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
"""Backbone pre-training: 3DGS as MCMC. The resulting checkpoint is the input
of scalable_train.py.
"""

import json
import os
import sys
import time
from argparse import ArgumentParser, Namespace
from random import randint

import torch
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from gode import evaluate
from scene import Scene, GaussianModel
from scene.gaussian_model import build_scaling_rotation
from utils.general_utils import safe_state
from utils.loss_utils import photometric_loss


def op_sigmoid(x, k=100, x0=0.995):
    return 1 / (1 + torch.exp(-k * (x - x0)))


def log(model_path, line):
    with open(os.path.join(model_path, "results.csv"), "a") as f:
        print(line, file=f)


def report(scene, model, model_path, iteration, eval_lpips=False, save_dir=None):
    psnr_v, ssim_v, lpips_v, fps_v, num = evaluate(
        scene, model, eval_lpips=eval_lpips, save_dir=save_dir)
    log(model_path, f"ITER={iteration},TEST,{psnr_v},{ssim_v},{lpips_v},{num},{fps_v}")
    print(f"ITER={iteration}: PSNR {psnr_v:.2f} SSIM {ssim_v:.3f} "
          f"LPIPS {lpips_v:.3f} FPS {fps_v:.1f} N={num}")


def training(args, dataset, opt, pipe):
    if dataset.cap_max == -1:
        sys.exit("Please specify the maximum number of Gaussians using --cap_max.")

    model = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, model, load_iteration=args.load_iter if args.eval_only else None)
    model.training_setup(opt)

    first_iter = 0
    if args.start_checkpoint:
        model_params, first_iter = torch.load(args.start_checkpoint)
        model.restore(model_params, opt)

    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                              dtype=torch.float32, device="cuda")
    model.pipe = pipe
    model.background = background

    if args.eval_only:
        report(scene, model, args.model_path, args.load_iter, eval_lpips=True)
        return

    cameras = []
    ema_loss = 0.0
    progress = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    start = time.time()

    for iteration in range(first_iter + 1, opt.iterations + 1):
        xyz_lr = model.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            model.oneupSHdegree()

        if not cameras:
            cameras = scene.getTrainCameras().copy()
        camera = cameras.pop(randint(0, len(cameras) - 1))

        if opt.random_background:
            model.background = torch.rand(3, device="cuda")
        image = model.render(camera)["render"]
        gt = camera.original_image.cuda()
        loss = photometric_loss(image, gt, opt.lambda_dssim)
        loss = loss + args.opacity_reg * torch.abs(model.get_opacity).mean()
        loss = loss + args.scale_reg * torch.abs(model.get_scaling).mean()
        loss.backward()

        with torch.no_grad():
            ema_loss = 0.4 * loss.item() + 0.6 * ema_loss
            if iteration % 10 == 0:
                progress.set_postfix({"Loss": f"{ema_loss:.7f}", "N": model.num_primitives})
                progress.update(10)

            if (opt.densify_from_iter < iteration < opt.densify_until_iter
                    and iteration % opt.densification_interval == 0):
                model.relocate_gs(dead_mask=(model.get_opacity <= 0.005).squeeze(-1))
                model.add_new_gs(cap_max=args.cap_max)

            if iteration < opt.iterations:
                model.optimizer.step()
                model.optimizer.zero_grad(set_to_none=True)

                # MCMC exploration noise, shaped by the local covariance.
                L = build_scaling_rotation(model.get_scaling, model.get_rotation)
                covariance = L @ L.transpose(1, 2)
                noise = (torch.randn_like(model._xyz)
                         * op_sigmoid(1 - model.get_opacity)
                         * args.noise_lr * xyz_lr)
                model._xyz.add_(torch.bmm(covariance, noise.unsqueeze(-1)).squeeze(-1))

            if args.test_every > 0 and iteration % args.test_every == 0:
                report(scene, model, args.model_path, iteration)

            if iteration in args.save_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            if iteration in args.checkpoint_iterations:
                torch.save((model.capture(), iteration),
                           os.path.join(args.model_path, f"chkpnt{iteration}.pth"))

    progress.close()
    train_time = time.time() - start
    log(args.model_path, f"TRAIN_TIME_SECONDS,{train_time:.2f}")
    print(f"Training time: {train_time:.2f}s")
    report(scene, model, args.model_path, opt.iterations, eval_lpips=True,
           save_dir=os.path.join(args.model_path, "renders"))


def parse_args():
    parser = ArgumentParser(description="Backbone pre-training (3DGS-MCMC)")
    model_params = ModelParams(parser)
    optimization_params = OptimizationParams(parser)
    pipeline_params = PipelineParams(parser)
    parser.add_argument("--config", type=str, default=None,
                        help="Per-scene JSON overrides, e.g. configs/bonsai.json")
    parser.add_argument("--test_every", type=int, default=5000)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true",
                        help="Evaluate a trained scene and exit")
    parser.add_argument("--load_iter", type=int, default=-1,
                        help="Iteration to load with --eval_only (-1 picks the latest)")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args(sys.argv[1:])
    if args.config:
        with open(args.config) as f:
            for key, value in json.load(f).items():
                setattr(args, key, value)
    return args, model_params, optimization_params, pipeline_params


if __name__ == "__main__":
    args, model_params, optimization_params, pipeline_params = parse_args()
    os.makedirs(args.model_path or "./output", exist_ok=True)
    print("Optimizing " + args.model_path)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as f:
        f.write(str(Namespace(**vars(args))))

    safe_state(args.quiet)
    training(args, model_params.extract(args), optimization_params.extract(args),
             pipeline_params.extract(args))
