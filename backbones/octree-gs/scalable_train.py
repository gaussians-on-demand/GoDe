"""GoDe on the Octree-GS backbone.

Takes a pre-trained scene, builds the LoD hierarchy, fine-tunes every level
jointly under quantization-aware training and writes the progressive bitstream.
"""

import logging
import os
import re
import sys
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
import yaml

from scene import Scene
from utils.general_utils import safe_state, parse_cfg
import gode


def parse_args():
    parser = ArgumentParser(description="GoDe progressive compression (Octree-GS)")
    parser.add_argument("--config", type=str, required=True,
                        help="Octree-GS YAML describing the model and the optimisation")
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Directory holding point_cloud.ply and the MLP checkpoints")
    parser.add_argument("--scene_name", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)

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
    group.add_argument("--appearance_reg_weight", type=float, default=0.0,
                       help="Weight of the backbone appearance prior: L1 on the anchor features")
    group.add_argument("--freeze_geometry_iters", type=int, default=3000,
                       help="Final iterations during which geometry is frozen")
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
    group.add_argument("--seed", type=int, default=0)
    group.add_argument("--quiet", action="store_true")

    args = parser.parse_args(sys.argv[1:])
    if args.hier_max is not None and not 0.0 < args.hier_max <= 1.0:
        parser.error("--hier_max must be in (0, 1].")
    if args.gradient_threshold < 0.0:
        parser.error("--gradient_threshold must be non-negative.")
    return args


def get_logger(path):
    """Octree-GS reports the octree construction through a logger."""
    logger = logging.getLogger("gode")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(os.path.join(path, "gode.log"))
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(message)s"))
        logger.addHandler(handler)
        logger.addHandler(logging.StreamHandler())
    return logger


def load_model(args, dataset, optimization, pipeline):
    """Instantiate the configured Octree-GS model and restore the checkpoint."""
    modules = __import__("scene.gs_model_" + dataset.base_model, fromlist=[""])
    config = dataset.model_config
    model = getattr(modules, config["name"])(**config["kwargs"])

    # Scene builds the octree from the point cloud first; the checkpoint is
    # then loaded over it, decoders included.
    scene = Scene(dataset, model, ply_path=args.checkpoint, shuffle=False,
                  resolution_scales=[1.0], logger=get_logger(args.model_path))
    model.load_ply(os.path.join(args.checkpoint, "point_cloud.ply"))
    model.load_mlp_checkpoints(args.checkpoint)
    model.set_coarse_interval(optimization)
    model.training_setup(optimization)

    model.pipe = pipeline
    model.background = scene.background
    model.render_mode = getattr(dataset, "render_mode", "rgb")
    model.train()
    return scene, model


def main():
    args = parse_args()
    with open(args.config) as handle:
        dataset, optimization, pipeline = parse_cfg(yaml.load(handle, Loader=yaml.FullLoader))

    dataset.source_path = args.source_path
    dataset.model_path = args.model_path
    dataset.scene_name = args.scene_name
    dataset.dataset_name = args.dataset_name
    dataset.eval = True

    # The core reads the schedule and the loss weights from one namespace.
    for name, value in vars(args).items():
        setattr(optimization, name, value)
    optimization.iterations = args.ft_iterations

    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as handle:
        handle.write(str(Namespace(**vars(args))))
    safe_state(args.quiet)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Output: {args.model_path}")
    print(f"Checkpoint: {args.checkpoint}")
    scene, model = load_model(args, dataset, optimization, pipeline)

    match = re.search(r"(?:ITER|iteration)[=_](\d+)", args.checkpoint, re.IGNORECASE)
    checkpoint_iter = int(match.group(1)) if match else 0
    rows = [gode.baseline_row(scene, model, checkpoint_iter)]
    gode.write_csv(args.model_path, rows)
    if args.baseline_eval_only:
        return

    if args.qat_warmup_iters > 0:
        gode.qat_warmup(scene, model, optimization, args.qat_warmup_iters)

    masks, _ = gode.build_hierarchy(scene, model, optimization)
    torch.save(masks, os.path.join(args.model_path, "masks.pt"))

    if args.pruning_only:
        if not getattr(model, "qa", False) and not args.no_qat:
            model.set_qa()
        gode.snapshot(scene, model, masks, optimization, iteration=0, train_time_s=0.0, rows=rows)
    else:
        gode.train_scalable(scene, model, masks, optimization, rows=rows)

    print(f"\nResults written to {gode.write_csv(args.model_path, rows)}")


if __name__ == "__main__":
    main()
