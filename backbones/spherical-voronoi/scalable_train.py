"""GoDe on the Spherical Voronoi backbone.

Takes a pre-trained scene, builds the LoD hierarchy, fine-tunes every level
jointly under quantization-aware training and writes the progressive bitstream.
"""

import os
import re
import sys
from argparse import ArgumentParser, Namespace

import torch

from arguments import ModelParams, OptimizationParams
from scene import Scene, BetaModel
from utils.general_utils import safe_state
from utils.system_utils import load_json
import gode

REPO = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = ArgumentParser(description="GoDe progressive compression (Spherical Voronoi)")
    ModelParams(parser)
    OptimizationParams(parser)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Pre-trained .pth or .ply produced by train.py")
    parser.add_argument("--color_rep", type=str, default="voronoi",
                        choices=["voronoi", "sh", "sb", "sgs"])
    parser.add_argument("--num_lobes", type=int, default=8)

    group = parser.add_argument_group("Hierarchy")
    group.add_argument("--num_levels", type=int, default=8,
                       help="Number of LoDs exposed for training and encoding")
    group.add_argument("--prune_levels", type=int, default=16,
                       help="Internal pruning steps (>= num_levels); the exposed levels are picked equidistantly")
    group.add_argument("--hier_min", type=float, default=None,
                       help="Coarsest level size, as a count or a fraction in (0, 1); default 10%% of the model")
    group.add_argument("--hier_max", type=float, default=None,
                       help="Finest level as a fraction of the model; by default it is inferred from --gradient_threshold")
    group.add_argument("--gradient_threshold", type=float, default=1e-12,
                       help="Normalized gradient below which a primitive is dropped from the finest level")
    group.add_argument("--prune_type", type=str, default="gradient_all",
                       choices=["gradient_all", "gradient_oneshot", "random"])

    group = parser.add_argument_group("Fine-tuning")
    group.add_argument("--ft_iterations", type=int, default=30_000)
    group.add_argument("--sample_level_ratio", type=float, default=1.0,
                       help="Level sampling ratio: 1 is uniform, >1 favours the finest level, <-1 the coarsest")
    group.add_argument("--appearance_reg_weight", type=float, default=0.00005,
                       help="Weight of the backbone appearance prior (0 disables it)")
    group.add_argument("--freeze_geometry_iters", type=int, default=3000,
                       help="Final iterations during which geometry is frozen")
    group.add_argument("--lr_start_iteration", type=int, default=0,
                       help="Iteration the learning-rate schedules resume from (0 restarts them)")
    group.add_argument("--qat_warmup_iters", type=int, default=0,
                       help="Iterations of quantization-aware warm-up on the full model before the hierarchy is built")
    group.add_argument("--no_qat", action="store_true",
                       help="Ablation: no quantization-aware training, float16 bitstream")
    group.add_argument("--pruning_only", action="store_true",
                       help="Ablation: skip fine-tuning, encode and evaluate the hierarchy as built")

    group = parser.add_argument_group("Reporting")
    group.add_argument("--encode_eval_iters", type=str, default="15000,30000",
                       help="Iterations at which a full encode/decode/evaluate snapshot is written")
    group.add_argument("--test_every", type=int, default=-1,
                       help="Evaluate every LoD each N iterations (-1 disables it)")
    group.add_argument("--dict_size", type=int, default=262_144,
                       help="Size in bytes of the zstd dictionary shared by the chunks")
    group.add_argument("--save_images", action="store_true",
                       help="Save the decoded test renders of the final snapshot")
    group.add_argument("--baseline_eval_only", action="store_true",
                       help="Evaluate the loaded checkpoint, write the baseline row and stop")
    group.add_argument("--quiet", action="store_true")

    args = parser.parse_args(sys.argv[1:])
    for key, value in load_json(args.config).items():
        setattr(args, key, value)

    if args.hier_max is not None and not 0.0 < args.hier_max <= 1.0:
        parser.error("--hier_max must be in (0, 1].")
    if args.gradient_threshold < 0.0:
        parser.error("--gradient_threshold must be non-negative.")

    args.cap_max = load_json(os.path.join(REPO, "config", "cap_max.json")).get(args.scene, 1_000_000)
    args.eval = True
    return args


def load_model(args):
    model = BetaModel(args.color_rep, num_lobes=args.num_lobes)
    scene = Scene(args, model, load_iteration=None)
    if args.checkpoint.endswith(".ply"):
        model.load_ply(args.checkpoint)
        model.training_setup(args)
    else:
        model.training_setup(args)
        model.restore(args.checkpoint, args)
    model.train()
    model.background = torch.tensor(
        [1, 1, 1] if args.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    return scene, model


def main():
    args = parse_args()
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as f:
        f.write(str(Namespace(**vars(args))))
    safe_state(args.quiet)

    # The checkpoint iteration only matters for the learning-rate schedules.
    match = re.search(r"(?:ITER|iteration)[=_](\d+)", args.checkpoint, re.IGNORECASE)
    checkpoint_iter = int(match.group(1)) if match else 0
    args.iterations = checkpoint_iter + args.ft_iterations

    print(f"Output: {args.model_path}")
    print(f"Checkpoint: {args.checkpoint}")
    scene, model = load_model(args)

    rows = [gode.baseline_row(scene, model, checkpoint_iter)]
    gode.write_csv(args.model_path, rows)
    if args.baseline_eval_only:
        return

    if args.qat_warmup_iters > 0:
        gode.qat_warmup(scene, model, args, args.qat_warmup_iters)

    masks, _ = gode.build_hierarchy(scene, model, args)
    torch.save(masks, os.path.join(args.model_path, "masks.pt"))

    if args.pruning_only:
        if not model.qa and not args.no_qat:
            model.set_qa()
        gode.snapshot(scene, model, masks, args, iteration=0, train_time_s=0.0, rows=rows)
    else:
        gode.train_scalable(scene, model, masks, args, rows=rows)

    print(f"\nResults written to {gode.write_csv(args.model_path, rows)}")


if __name__ == "__main__":
    main()
