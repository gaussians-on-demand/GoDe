"""GoDe on the Scaffold-GS backbone.

Takes a pre-trained scene, builds the LoD hierarchy over the anchors,
fine-tunes every level jointly under quantization-aware training and writes the
progressive bitstream. The neural decoders are global: they are stored next to
the bitstream by Scene.save(), not inside the LoD chunks.
"""

import os
import random
import re
import sys
from argparse import ArgumentParser, Namespace

import numpy as np
import torch

from arguments import ModelParams, OptimizationParams, PipelineParams
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import gode


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def parse_args():
    parser = ArgumentParser(description="GoDe progressive compression (Scaffold-GS)")
    model_params = ModelParams(parser)
    OptimizationParams(parser)
    pipeline_params = PipelineParams(parser)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Pre-trained point_cloud/iteration_N directory produced by train.py")

    group = parser.add_argument_group("Hierarchy")
    group.add_argument("--num_levels", type=int, default=8,
                       help="Number of LoDs exposed for training and encoding")
    group.add_argument("--prune_levels", type=int, default=16,
                       help="Internal pruning steps (>= num_levels); the exposed levels are picked equidistantly")
    group.add_argument("--hier_min", type=float, default=None,
                       help="Coarsest level size, as a count or a fraction in (0, 1); default 10%% of the anchors")
    group.add_argument("--hier_max", type=float, default=None,
                       help="Finest level as a fraction of the anchors; by default it is inferred from --gradient_threshold")
    group.add_argument("--gradient_threshold", type=float, default=1e-12,
                       help="Normalized gradient below which an anchor is dropped from the finest level")
    group.add_argument("--prune_type", type=str, default="gradient_all",
                       choices=["gradient_all", "gradient_oneshot", "random"])

    group = parser.add_argument_group("Fine-tuning")
    group.add_argument("--ft_iterations", type=int, default=30_000)
    group.add_argument("--sample_level_ratio", type=float, default=1.0,
                       help="Level sampling ratio: 1 is uniform, >1 favours the finest level, <-1 the coarsest")
    group.add_argument("--appearance_reg_weight", type=float, default=0.0,
                       help="L1 shrinkage of the anchor features (0 disables it)")
    group.add_argument("--freeze_geometry_iters", type=int, default=3000,
                       help="Final iterations during which anchor geometry is frozen")
    group.add_argument("--lr_start_iteration", type=int, default=None,
                       help="Iteration the learning-rate schedules resume from; "
                            "defaults to the iteration of the loaded checkpoint")
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
    group.add_argument("--baseline_eval_only", action="store_true",
                       help="Evaluate the loaded checkpoint, write the baseline row and stop")
    group.add_argument("--save_images", action="store_true",
                       help="Save the decoded test renders of the final snapshot")
    group.add_argument("--seed", type=int, default=0)
    group.add_argument("--quiet", action="store_true")

    args = parser.parse_args(sys.argv[1:])
    if args.hier_max is not None and not 0.0 < args.hier_max <= 1.0:
        parser.error("--hier_max must be in (0, 1].")
    if args.gradient_threshold < 0.0:
        parser.error("--gradient_threshold must be non-negative.")

    if args.lr_start_iteration is None:
        match = re.search(r"iteration_(\d+)", args.checkpoint)
        args.lr_start_iteration = int(match.group(1)) if match else 0

    args.eval = True
    args.iterations = args.lr_start_iteration + args.ft_iterations
    return args, model_params, pipeline_params


def load_model(args, dataset, pipe):
    model = GaussianModel(
        dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth,
        dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank,
        dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist,
        dataset.add_cov_dist, dataset.add_color_dist, quantization=not args.no_qat,
    )
    scene = Scene(dataset, model, shuffle=False)

    checkpoint = args.checkpoint
    if checkpoint.endswith(".ply"):
        checkpoint = os.path.dirname(checkpoint)
    model.load_ply_sparse_gaussian(os.path.join(checkpoint, "point_cloud.ply"))
    model.load_mlp_checkpoints(checkpoint)
    model.training_setup(args)

    model.pipe = pipe
    model.background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    return scene, model


def main():
    args, model_params, pipeline_params = parse_args()
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as f:
        f.write(str(Namespace(**vars(args))))
    safe_state(args.quiet)
    set_seed(args.seed)

    print(f"Output: {args.model_path}")
    print(f"Checkpoint: {args.checkpoint}")
    scene, model = load_model(args, model_params.extract(args), pipeline_params.extract(args))

    checkpoint_iter = int(match.group(1)) if (match := re.search(
        r"(?:ITER|iteration)[=_](\d+)", args.checkpoint, re.IGNORECASE)) else 0
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
