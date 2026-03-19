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
import json
import math
from lpips import lpips
import copy
import os
import time
import numpy as np
import torch
from PIL import Image
from random import randint
from train import prepare_output_and_logger, training_report
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import torch.nn as nn
import random 
import numpy as np



def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
           

def geometric_progression(tmin, tmax, L):
    b = np.exp((np.log(tmax) - np.log(tmin)) / (L - 1))
    print(b)
    arr = np.array([i for i in range(L)])
    return ((b ** arr) * tmin).astype(np.int32)
    
            
def write(string):
    with open(os.path.join(args.model_path, 'results.csv'), 'a') as f:
        print(string, file=f)

    
def accumulate_gradients(scene, opt, gaussians, pipe, background, mask, render_fun):
    print('Accumulating gradients...')
    for viewpoint_cam in scene.getTrainCameras():

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render_fun(viewpoint_cam, gaussians, pipe, bg, mask=mask)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward(retain_graph=True)
    print('Done')
    
    
def fine_tuning(scene, opt, gaussians, pipe, background, mask, first_iter, iterations, render_fun):

    progress_bar = tqdm(range(first_iter, first_iter+opt.iterations), desc="Training progress")
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    for iteration in range(first_iter, first_iter + iterations+1):        

        gaussians.update_learning_rate(iteration)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render_fun(viewpoint_cam, gaussians, pipe, bg, mask=mask)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        loss.backward()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "num_gaussians": gaussians._xyz.shape[0]})
                progress_bar.update(10)
            if iteration == opt.iterations+1:
                progress_bar.close()

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)


def iterative_masking(dataset, opt, pipe, testing_iterations, infer_max=False):
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    
    gaussians.load_ply(os.path.join(args.pretrained_dir, 'point_cloud', f'iteration_{args.load_iter}', 'point_cloud.ply'))
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    masks = []
    iteration = 1
    
    k = 0
    keep = True
    num_gaussians = gaussians._xyz.shape[0]
    
    print(args.min, args.num_levels)

    mask = None
    render_fun =  render

    k_s = list(np.flip((num_gaussians - geometric_progression(args.min, num_gaussians*args.max, args.num_levels))))

    print(k_s)
    for i in range(args.num_levels):
        if keep:
            accumulate_gradients(scene, opt, gaussians, pipe, background, mask, render_fun)
            
        num_gaussians = num_gaussians if mask is None else mask.sum()
        k = k_s[i] 
        if i > 0:
            k -= k_s[0]
        
        mask, pruned_indices = gaussians.gradient_prune(k, prune_type=args.prune_type)
        
        if i == 0:
            with torch.no_grad():       
                scene.gaussians._xyz = scene.gaussians._xyz[mask].clone().detach().requires_grad_(True)
                scene.gaussians._opacity = scene.gaussians._opacity[mask].clone().detach().requires_grad_(True)
                scene.gaussians._rotation = scene.gaussians._rotation[mask].clone().detach().requires_grad_(True)
                scene.gaussians._scaling = scene.gaussians._scaling[mask].clone().detach().requires_grad_(True)
                scene.gaussians._features_dc = scene.gaussians._features_dc[mask].clone().detach().requires_grad_(True)
                scene.gaussians._features_rest = scene.gaussians._features_rest[mask].clone().detach().requires_grad_(True)
                mask = torch.ones(mask.sum().item(), device='cuda').bool()
                gaussians.training_setup(opt)
                
        keep = True # iterative
        masks.append(mask.cuda())
        test(None, iteration, None, None, l1_loss, None, testing_iterations, scene, render_fun, (pipe, background), dump_images=False, eval_lpips=False, mask=mask)
        if keep:
            gaussians.optimizer.zero_grad(set_to_none=True)
    
    for mask in masks:
        test(None, iteration, None, None, l1_loss, None, testing_iterations, scene, render_fun, (pipe, background), dump_images=False, eval_lpips=False, mask=mask)
    
    
    gaussians.optimizer.zero_grad(set_to_none=True)
    
    torch.save(masks, os.path.join(args.model_path, 'masks.pt'))
    
    return scene, masks, pruned_indices
    

def train(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, scene, masks, load_iter=None):
    first_iter = 0
    prepare_output_and_logger(dataset)
    gaussians = scene.gaussians
    gaussians.training_setup(opt)
    
    gaussians.set_qa()

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, first_iter+opt.iterations), desc="Training progress")
    
    testing_iterations = list(map(lambda x: x + first_iter, testing_iterations))
    
    
    if args.weighted_sampling:
        progr = np.flip(geometric_progression(1, args.G, len(masks)))
        probabilities = torch.tensor(progr / progr.sum(), dtype=torch.float32)
    else:
        probabilities = torch.tensor([1 for i in range(len(masks))], dtype=torch.float32)
    
    indices = torch.multinomial(probabilities, opt.iterations, replacement=True)

    render_fun = render

    first_iter += 1
    start_time = time.time()
    for iteration in range(first_iter, first_iter+opt.iterations + 1):        
        
        m_index = indices[iteration-first_iter-1]
        mask = masks[m_index]
        
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render_fun(viewpoint_cam, gaussians, pipe, bg, mask=mask)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        # if args.reduced or args.qa:
        loss = loss + 0.05 * gaussians._features_rest[mask].abs().mean()
        
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "num_gaussians": gaussians._xyz.shape[0]})
                progress_bar.update(10)
            if iteration == opt.iterations+first_iter:
                progress_bar.close()

            if iteration in testing_iterations:
                
                for i, mask in enumerate(masks):
                    gaussians = scene.gaussians
                    gaussians_cpy = copy.deepcopy(gaussians)
                    scene.gaussians = gaussians_cpy
                    scene.gaussians._xyz = scene.gaussians._xyz[mask]
                    scene.gaussians._opacity = scene.gaussians._opacity[mask]
                    scene.gaussians._rotation = scene.gaussians._rotation[mask]
                    scene.gaussians._scaling = scene.gaussians._scaling[mask]
                    scene.gaussians._features_dc = scene.gaussians._features_dc[mask]
                    scene.gaussians._features_rest = scene.gaussians._features_rest[mask]
                    
                    # if iteration == testing_iterations[-1] and not args.test_only:
                    #     eval_lpips = True 
                    #     dump_images = True
                    
                    # test(None, iteration, None, None, l1_loss, None, testing_iterations, scene, render_fun, (pipe, background), 
                    #      dump_images=dump_images, eval_lpips=eval_lpips, mask=None, level=args.num_levels-i-1)
                    
                    scene.gaussians = gaussians
            
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)
    
    train_time = time.time() - start_time   
    write(f"train_time: {train_time}")
    
    with torch.no_grad():
        for i, mask in enumerate(masks):
            gaussians = scene.gaussians
            gaussians_cpy = copy.deepcopy(gaussians)
            scene.gaussians = gaussians_cpy
            scene.gaussians._xyz = scene.gaussians._xyz[mask]
            scene.gaussians._opacity = scene.gaussians._opacity[mask]
            scene.gaussians._rotation = scene.gaussians._rotation[mask]
            scene.gaussians._scaling = scene.gaussians._scaling[mask]
            scene.gaussians._features_dc = scene.gaussians._features_dc[mask]
            scene.gaussians._features_rest = scene.gaussians._features_rest[mask]
            
            # if iteration == testing_iterations[-1] and not args.test_only:
            #     eval_lpips = True 
            #     dump_images = True
            
            test(None, iteration, None, None, l1_loss, None, testing_iterations, scene, render_fun, (pipe, background), 
                dump_images=False, eval_lpips=True, mask=None, level=args.num_levels-i-1)
            
            scene.gaussians = gaussians
    
    
    with torch.no_grad():
        disjointed_masks = [masks[-1]]
        
        for i in range(args.num_levels-1, 0, -1):
            disjointed_masks.append(torch.logical_xor(masks[i].cuda(), masks[i-1].cuda()))

        order = gaussians._sort_morton()
        disjointed_masks = [m[order] for m in disjointed_masks]
        dict_save_path = f'{args.model_path}/zstd_dict.dict'
        zstd_dict = gaussians.create_zstd_dictionary_from_quantized_gaussians(save_path=dict_save_path)
        
        
    sizes = []
    with torch.no_grad():
        for i, mask in enumerate(disjointed_masks):
            gaussians = scene.gaussians
            gaussians_cpy = copy.deepcopy(gaussians)
            scene.gaussians = gaussians_cpy  
            scene.gaussians._xyz = scene.gaussians._xyz[mask]
            scene.gaussians._opacity = scene.gaussians._opacity[mask]
            scene.gaussians._scaling = scene.gaussians._scaling[mask]
            scene.gaussians._rotation = scene.gaussians._rotation[mask]
            scene.gaussians._features_dc = scene.gaussians._features_dc[mask]
            scene.gaussians._features_rest = scene.gaussians._features_rest[mask]
            start = time.time()
            size = scene.gaussians.save_zstd(f"{args.model_path}/data_{i}.zst", zstd_dict=zstd_dict, compression_level=22)
            end = time.time() - start
            write(f'Enc time for Lod_{i},{end}, Size,{size}')
            print(f'Enc time for Lod_{i},{end}, Size,{size}')
            sizes.append(size)
            scene.gaussians = gaussians
            
        dec_time = 0.0
        with torch.no_grad():

            def _to_dev(x, dev):
                return x if x.device == dev else x.to(dev)

            def _concat_gaussians_(dst, src):
                dev = dst._xyz.device
                dst._xyz          = torch.cat([dst._xyz,           _to_dev(src._xyz, dev)], dim=0)
                dst._opacity       = torch.cat([dst._opacity,       _to_dev(src._opacity, dev)], dim=0)
                dst._scaling   = torch.cat([dst._scaling,   _to_dev(src._scaling, dev)], dim=0)
                dst._rotation   = torch.cat([dst._rotation,   _to_dev(src._rotation, dev)], dim=0)
                dst._features_dc   = torch.cat([dst._features_dc,   _to_dev(src._features_dc, dev)], dim=0)
                dst._features_rest   = torch.cat([dst._features_rest,   _to_dev(src._features_rest, dev)], dim=0)

            acc = copy.deepcopy(scene.gaussians)        
            t0 = time.time()
            acc.load_zstd(os.path.join(args.model_path, f"data_0.zst"), dict_save_path)
            dec_time += time.time() - t0
            write(f'Decoding time for LoD0,{dec_time}, total,{dec_time}')
            print(f'Decoding time for LoD0,{dec_time}, total,{dec_time}')

            scene.gaussians = acc
            test(None, iteration, None, None, l1_loss, None, testing_iterations, scene, render_fun, (pipe, background), 
                    dump_images=False, eval_lpips=False, mask=None, level=i, size=acc._xyz.shape[0])

            num_lods = len(disjointed_masks)  
            for i in range(1, num_lods):
                tmp = copy.deepcopy(scene.gaussians)    
                t0 = time.time()
                tmp.load_zstd(os.path.join(args.model_path, f"data_{i}.zst"), dict_save_path)
                _concat_gaussians_(acc, tmp)
                dt = time.time() - t0
                write(f'Decoding time for LoD{i},{dt}, total,{dec_time}')
                print(f'Decoding time for LoD{i},{dt}, total,{dec_time}')
                dec_time += dt

                scene.gaussians = acc
                test(None, iteration, None, None, l1_loss, None, testing_iterations, scene, render_fun, (pipe, background), 
                    dump_images=False, eval_lpips=False, mask=None, level=i, size=acc._xyz.shape[0])

            
@torch.no_grad()
def test(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, moe=None, 
         gate=None, distance=None, dump_images=False, eval_lpips=False, mask=None, level=None, size=None):
    torch.cuda.empty_cache()
    validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                            {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
    result_dict = {}
    if level is not None:
        os.makedirs(os.path.join(args.model_path, 'renders', f'L{level}'), exist_ok=True)
    

    with torch.no_grad():
        if mask is not None:
            gaussians = scene.gaussians
            gaussians_cpy = copy.deepcopy(gaussians)
            scene.gaussians = gaussians_cpy
            scene.gaussians._xyz = scene.gaussians._xyz[mask]
            scene.gaussians._opacity = scene.gaussians._opacity[mask]
            scene.gaussians._rotation = scene.gaussians._rotation[mask]
            scene.gaussians._scaling = scene.gaussians._scaling[mask]
            scene.gaussians._features_dc = scene.gaussians._features_dc[mask]
            scene.gaussians._features_rest = scene.gaussians._features_rest[mask]

    for config in validation_configs:
        if config['name'] == 'train': 
            continue
        if config['cameras'] and len(config['cameras']) > 0:
            l1_test = 0.0
            psnr_test = 0.0
            ssim_test = 0.0
            lpips_test = 0.0
            fps_test = 0.0
            for idx, viewpoint in enumerate(config['cameras']):
 
                pkg = renderFunc(viewpoint, scene.gaussians, measure_fps=True, *renderArgs)
                fps = pkg['fps']
                image = torch.clamp(pkg["render"], 0.0, 1.0)
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                if dump_images and config['name'] == 'test':
                    np_img = (np.transpose(image.cpu().numpy(), (1, 2, 0)) * 255).astype(np.uint8)
                    img = Image.fromarray(np_img)
                    img.save(os.path.join(args.model_path, 'renders', f'L{level}', f'{idx}.png'))

                l1_test += l1_loss(image, gt_image).mean().double()
                psnr_test += psnr(image, gt_image).mean().double()
                ssim_test += ssim(image, gt_image).mean().double()
                fps_test += fps
                
                if eval_lpips:
                    lpips_test += lpips_loss(image, gt_image, normalize=True).item()
                    
            psnr_test /= len(config['cameras'])
            ssim_test /= len(config['cameras'])
            lpips_test /= len(config['cameras'])
            l1_test /= len(config['cameras'])   
            fps_test /= len(config['cameras'])
            
            result_dict[config['name']] = psnr_test
            
            num_gaussians = mask.sum().item() if mask is not None else scene.gaussians._xyz.shape[0]
            if config['name'] == 'test':
                
                write(f'{iteration}, {psnr_test}, {ssim_test}, {lpips_test}, {num_gaussians}, {size}, {fps}')   
                #wandb.log({'psnr': psnr_test, 'ssim': ssim_test, 'lpips':lpips, 'num_gaussians': num_gaussians})    
                print("\n[ITER {}], PSNR {}, SSIM {}, FPS {}, #W {}".format(iteration, psnr_test, ssim_test, fps_test, num_gaussians))
    if mask is not None:
        scene.gaussians = gaussians
    torch.cuda.empty_cache()
    return result_dict


if __name__ == "__main__":
    
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000, 2000, 3000, 5000, 7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[60_000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--num_levels", type=int, default=8, help='number of levels')
    parser.add_argument("--min", type=int, default=100_000, help='minimum number of gaussians of the lowest lod')
    parser.add_argument("--max", type=float, default=0.75, help="maximum number of gaussians of the highest level (0, 1]")
    parser.add_argument("--weighted_sampling", action='store_true', default=False, help='wheter if you want to prioritize higher LoDs')
    parser.add_argument("--G", type=int, default=5, help='weghted sampling weights. the higher, the more higher LoD will be sampled during fine-tuning')
    parser.add_argument("--test_only", action='store_true', default=True)
    parser.add_argument("--pretrained_dir", type=str, required=True, help='root containing pre-trained model')
    parser.add_argument("--load_iter", type=int, default=30000)
    parser.add_argument("--prune_type", type=str, default='gradient_all')
    parser.add_argument("--dict_size", type=int, default=112_640, help="Size of the shared Zstd dictionary in bytes")

    
    args = parser.parse_args(sys.argv[1:])
    
    args.eval = True
    args.test_iterations = [30_000]
    

    print("Optimizing " + args.model_path)
    print(args.source_path)

    print('cuda available: ')
    print(torch.cuda.is_available())
    
    print('current gpu: ', torch.cuda.current_device())

    # Initialize system state (RNG)
    safe_state(args.quiet)
    lpips_loss = lpips.LPIPS(net='vgg').to('cuda')
    
    scene, masks, pruned_indices = iterative_masking(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations)
    train(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, scene, masks)

    # All done
    print("\nTraining complete.")
