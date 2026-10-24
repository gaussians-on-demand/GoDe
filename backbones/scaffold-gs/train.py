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
"""Backbone pre-training: Scaffold-GS anchors with neural decoders. The
resulting checkpoint is the input of scalable_train.py.
"""

import os
import sys
import time
from argparse import ArgumentParser, Namespace
from random import randint

import torch
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import prefilter_voxel, render
from gode import evaluate
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.loss_utils import photometric_loss


def log(model_path, line):
    with open(os.path.join(model_path, "results.csv"), "a") as f:
        print(line, file=f)


def report(scene, model, model_path, iteration, eval_lpips=False, save_dir=None):
    psnr_v, ssim_v, lpips_v, fps_v, num = evaluate(
        scene, model, eval_lpips=eval_lpips, save_dir=save_dir)
    log(model_path, f"ITER={iteration},TEST,{psnr_v},{ssim_v},{lpips_v},{num},{fps_v}")
    print(f"ITER={iteration}: PSNR {psnr_v:.2f} SSIM {ssim_v:.3f} "
          f"LPIPS {lpips_v:.3f} FPS {fps_v:.1f} anchors={num}")
    model.train()


def training(args, dataset, opt, pipe):
    model = GaussianModel(
        dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth,
        dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank,
        dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist,
        dataset.add_cov_dist, dataset.add_color_dist)
    scene = Scene(dataset, model, shuffle=False)
    model.training_setup(opt)

    first_iter = 0
    if args.start_checkpoint:
        model_params, first_iter = torch.load(args.start_checkpoint)
        model.restore(model_params, opt)

    model.pipe = pipe
    model.background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                                    dtype=torch.float32, device="cuda")

    cameras = []
    ema_loss = 0.0
    model.train()
    progress = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    start = time.time()

    for iteration in range(first_iter + 1, opt.iterations + 1):
        model.update_learning_rate(iteration)

        if not cameras:
            cameras = scene.getTrainCameras().copy()
        camera = cameras.pop(randint(0, len(cameras) - 1))

        visible = prefilter_voxel(camera, model, pipe, model.background)
        retain_grad = iteration < opt.update_until
        package = render(camera, model, pipe, model.background,
                         visible_mask=visible, retain_grad=retain_grad)
        image, scaling = package["render"], package["scaling"]

        gt = camera.original_image.cuda()
        # The scaling regulariser keeps the neural Gaussians compact.
        loss = photometric_loss(image, gt, opt.lambda_dssim) + 0.01 * scaling.prod(dim=1).mean()
        loss.backward()

        with torch.no_grad():
            ema_loss = 0.4 * loss.item() + 0.6 * ema_loss
            if iteration % 10 == 0:
                progress.set_postfix({"Loss": f"{ema_loss:.7f}", "anchors": model.num_primitives})
                progress.update(10)

            # Anchor growing and pruning.
            if opt.start_stat < iteration < opt.update_until:
                model.training_statis(package["viewspace_points"], package["neural_opacity"],
                                      package["visibility_filter"], package["selection_mask"],
                                      visible)
                if iteration > opt.update_from and iteration % opt.update_interval == 0:
                    model.adjust_anchor(check_interval=opt.update_interval,
                                        success_threshold=opt.success_threshold,
                                        grad_threshold=opt.densify_grad_threshold,
                                        min_opacity=opt.min_opacity)
            elif iteration == opt.update_until:
                del model.opacity_accum, model.offset_gradient_accum, model.offset_denom
                torch.cuda.empty_cache()

            if iteration < opt.iterations:
                model.optimizer.step()
                model.optimizer.zero_grad(set_to_none=True)

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
    parser = ArgumentParser(description="Backbone pre-training (Scaffold-GS)")
    model_params = ModelParams(parser)
    optimization_params = OptimizationParams(parser)
    pipeline_params = PipelineParams(parser)
    parser.add_argument("--test_every", type=int, default=5000)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(sys.argv[1:]), model_params, optimization_params, pipeline_params


if __name__ == "__main__":
    args, model_params, optimization_params, pipeline_params = parse_args()
    os.makedirs(args.model_path, exist_ok=True)
    print("Optimizing " + args.model_path)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as f:
        f.write(str(Namespace(**vars(args))))

    safe_state(args.quiet)
    training(args, model_params.extract(args), optimization_params.extract(args),
             pipeline_params.extract(args))
