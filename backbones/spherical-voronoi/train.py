#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
"""Backbone pre-training: MCMC-style densification with a Spherical Voronoi
appearance model. The resulting checkpoint is the input of scalable_train.py.
"""

import os
import sys
import time
from argparse import ArgumentParser, Namespace
from random import randint

import torch
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams
from scene import Scene, BetaModel
from scene.beta_model import build_scaling_rotation
from utils.general_utils import safe_state
from utils.loss_utils import photometric_loss
from utils.system_utils import load_json

REPO = os.path.dirname(os.path.abspath(__file__))


def log(model_path, line):
    with open(os.path.join(model_path, "results.csv"), "a") as f:
        print(line, file=f)


def evaluate(scene, args, iteration, splits=("test",), **kwargs):
    for split in splits:
        psnr_v, ssim_v, lpips_v = scene.eval(split, iteration=iteration, **kwargs)
        log(args.model_path, f"ITER={iteration},{split.upper()},{psnr_v},{ssim_v},{lpips_v}")
        print(f"ITER={iteration}: {split.upper()} PSNR {psnr_v:.2f} SSIM {ssim_v:.3f} LPIPS {lpips_v:.3f}")


def training(args):
    model = BetaModel(args.color_rep, num_lobes=args.num_lobes)
    scene = Scene(args, model, load_iteration=None)
    model.training_setup(args)
    if args.start_checkpoint:
        model.restore(args.start_checkpoint, args)

    background = torch.tensor([1, 1, 1] if args.white_background else [0, 0, 0],
                              dtype=torch.float32, device="cuda")
    model.background = torch.rand(3, device="cuda") if args.random_background else background

    if args.eval_only:
        evaluate(scene, args, args.iterations, dump_images=True, eval_lpips=True)
        return

    if args.cap_max < model._xyz.shape[0]:
        print(f"cap_max ({args.cap_max}) < initial points ({model._xyz.shape[0]}): subsampling.")
        model.subsample_to(args.cap_max)

    # The Voronoi sites and their temperatures keep learning after the rest of
    # the model is frozen, which stabilises the final appearance.
    freeze_at = args.iterations - args.freeze_iters
    frozen = False

    cameras = []
    ema_loss = 0.0
    model.train()
    progress = tqdm(range(1, args.iterations + 1), desc="Training progress")
    start = time.time()

    for iteration in range(1, args.iterations + 1):
        xyz_lr = model.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            model.oneupSHdegree()

        if not cameras:
            cameras = scene.getTrainCameras().copy()
        camera = cameras.pop(randint(0, len(cameras) - 1))

        image = model.render(camera)["render"]
        gt = camera.original_image.cuda()
        loss = photometric_loss(image, gt, args.lambda_dssim)
        if args.densify_from_iter < iteration < args.densify_until_iter:
            loss += args.opacity_reg * torch.abs(model.get_opacity).mean()
            loss += args.scale_reg * torch.abs(model.get_scaling).mean()
        loss.backward()

        with torch.no_grad():
            ema_loss = 0.4 * loss.item() + 0.6 * ema_loss
            if iteration % 10 == 0:
                progress.set_postfix({"Loss": f"{ema_loss:.4f}",
                                      "Beta": f"{model._beta.mean().item():.2f}"})
                progress.update(10)

            if not frozen and model.color_representation == "voronoi" and iteration > freeze_at:
                for group in model.optimizer.param_groups:
                    if group["name"] not in ("colors", "log_tau_sites"):
                        group["lr"] = 0.0
                frozen = True

            model.optimizer.step()
            model.optimizer.zero_grad(set_to_none=True)

            if (args.densify_from_iter < iteration < args.densify_until_iter
                    and iteration % args.densification_interval == 0):
                model.relocate_gs(dead_mask=(model.get_opacity <= 0.005).squeeze(-1))
                model.add_new_gs(cap_max=args.cap_max)

                # MCMC exploration noise, shaped by the local covariance.
                L = build_scaling_rotation(model.get_scaling, model.get_rotation)
                covariance = L @ L.transpose(1, 2)
                noise = (torch.randn_like(model._xyz)
                         * torch.pow(1 - model.get_opacity, 100)
                         * args.noise_lr * xyz_lr)
                model._xyz.add_(torch.bmm(covariance, noise.unsqueeze(-1)).squeeze(-1))

            if args.test_every > 0 and iteration % args.test_every == 0:
                evaluate(scene, args, iteration, splits=("test", "train"))

            if iteration in args.save_iterations:
                print(f"\n[ITER {iteration}] Saving scene")
                scene.save(iteration)

            if iteration in args.checkpoint_iterations:
                model.save(os.path.join(args.model_path, f"ckpts_ITER={iteration}.pth"))

    progress.close()
    train_time = time.time() - start
    log(args.model_path, f"TRAIN_TIME_SECONDS,{train_time:.2f}")
    print(f"Training time: {train_time:.2f}s")
    evaluate(scene, args, args.iterations, dump_images=True, eval_lpips=True)


def parse_args():
    parser = ArgumentParser(description="Backbone pre-training")
    ModelParams(parser)
    OptimizationParams(parser)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--color_rep", type=str, default="voronoi",
                        choices=["voronoi", "sh", "sb", "sgs"])
    parser.add_argument("--num_lobes", type=int, default=8)
    parser.add_argument("--freeze_iters", type=int, default=3000,
                        help="Final iterations during which only colors and temperatures keep learning")
    parser.add_argument("--test_every", type=int, default=1000)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true",
                        help="Evaluate the loaded checkpoint and exit")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args(sys.argv[1:])
    for key, value in load_json(args.config).items():
        setattr(args, key, value)

    args.cap_max = load_json(os.path.join(REPO, "config", "cap_max.json"))[args.scene]
    args.eval = True
    args.save_iterations = args.save_iterations or [args.iterations]
    return args


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as f:
        f.write(str(Namespace(**vars(args))))

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    training(args)
