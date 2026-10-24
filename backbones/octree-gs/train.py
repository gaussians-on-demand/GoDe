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

import os
import shutil
import numpy as np

import subprocess

cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')

import csv
import torch
import torchvision
import json
import wandb
import time
from datetime import datetime
from os import makedirs
import shutil
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
import lpips
from random import randint, random
from utils.loss_utils import l1_loss, ssim
import sys
from gaussian_renderer import network_gui
from scene import Scene
from utils.general_utils import safe_state, parse_cfg, get_render_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
import yaml
import warnings
warnings.filterwarnings('ignore')


lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

stat_dict = {
    'psnr_train': {},
    'psnr_test': {},
    'ssim_train': {},
    'ssim_test': {},
    'fps_train': {},
    'fps_test': {},
    'lpips_train': {},
    'lpips_test': {},
    'active_gaussians_train': {},
    'active_gaussians_test': {},
    'size': [],
    'num_gaussians': []
}


def update_dict(dataset, scale, psnr_test, ssim_test, lpips_test, fps, active_gaussians, num_gaussians):
    # Mappa chiavi del dataset ai valori
    data_map = {
        'psnr': psnr_test,
        'ssim': ssim_test,
        'lpips': lpips_test,
        'fps': fps,
        'active_gaussians': active_gaussians
    }
    
    for key, values in data_map.items():
        dict_key = f"{key}_{dataset}"
        if dict_key in stat_dict:
            if scale not in stat_dict[dict_key]:
                stat_dict[dict_key][scale] = []
            stat_dict[dict_key][scale].append(values)
    
    stat_dict['num_gaussians'].append(num_gaussians)


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = False
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")


def accumulate_gradients(scene, opt, gaussians, pipe, render_fun, dataset, mask, scale, iteration=30_000):
    cams = scene.getTrainCameras(scale)
    for viewpoint_cam in cams:
        render_pkg = render_fun(viewpoint_cam, gaussians, pipe, scene.background, iteration, dataset.render_mode, mask=mask, disable_lod=True)
        image = render_pkg["render"]
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_loss = (1.0 - ssim(image, gt_image))
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss
        loss.backward(retain_graph=True)


def build_lod_model(scene, opt, pipe, render_fun, dataset, lod_levels=5):
    accumulate_gradients(scene, opt, scene.gaussians, pipe, render_fun, dataset, None, 1.0)
    u_mask = (scene.gaussians._anchor.grad.sum(dim=-1) != 0)
    num_gaussians = u_mask.sum().item()
    scene.gaussians.optimizer.zero_grad(set_to_none=True)
    
    scene.gaussians._opacity = scene.gaussians._opacity[u_mask].clone().detach().requires_grad_(True)
    scene.gaussians._anchor = scene.gaussians._anchor[u_mask].clone().detach().requires_grad_(True)
    scene.gaussians._offset = scene.gaussians._offset[u_mask].clone().detach().requires_grad_(True)
    scene.gaussians._scaling = scene.gaussians._scaling[u_mask].clone().detach().requires_grad_(True)
    scene.gaussians._rotation = scene.gaussians._rotation[u_mask].clone().detach().requires_grad_(True)
    scene.gaussians._features_rest = scene.gaussians._features_rest[u_mask].clone().detach().requires_grad_(True)
    scene.gaussians._features_dc = scene.gaussians._features_dc[u_mask].clone().detach().requires_grad_(True)
    scene.gaussians.training_setup(opt)
    
    k_s = num_gaussians - np.array(([num_gaussians//2**i for i in range(0, lod_levels+1)]))
    
    mask = None
    masks = []
    for k in k_s:    
        accumulate_gradients(scene, opt, scene.gaussians, pipe, render_fun, dataset, mask, 1.0)
        num_gaussians = num_gaussians if mask is None else mask.sum()
        mask, _ = scene.gaussians.gradient_prune(k, u_mask=mask)
        scene.gaussians.optimizer.zero_grad(set_to_none=True)
        masks.append(mask)
    
    with torch.no_grad():
        scene.gaussians.build_lod_representation(masks)
    

def training(dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, wandb=None, logger=None, ply_path=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    modules = __import__('scene.gs_model_'+dataset.base_model, fromlist=[''])
    model_config = dataset.model_config
    gaussians = getattr(modules, model_config['name'])(**model_config['kwargs'])
    scene = Scene(dataset, gaussians, ply_path=ply_path, shuffle=False, logger=logger, resolution_scales=dataset.resolution_scales)
    
    if ply_path is not None:
        gaussians.load_ply(ply_path)
    
    # if args.qat:
    #     gaussians.set_qa()
    gaussians.set_coarse_interval(opt)
    gaussians.training_setup(opt)
    
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    modules = __import__('gaussian_renderer')
    
    
    render_fun = getattr(modules, get_render_func(dataset.base_model))
    
    # if lod_gradient_based:
    #     build_lod_model(scene, opt, pipe, render_fun, dataset)
    #     gaussians.training_setup(opt)
    #     first_iter = 25_000
    
    scale_weights = torch.tensor([2**i for i in range(len(dataset.resolution_scales) -1, -1, -1)], dtype=torch.float)
    scale_indices = torch.multinomial(scale_weights, opt.iterations, replacement=True)
    
    
    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if not args.ms_train or iteration <= 25_000:
            scale = 1.0
        else:
            scale = dataset.resolution_scales[scale_indices[iteration-1]]
            # if random() < 0.75:
            #     scale = 1.0

        # Pick a random Camera
        # if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras(scale).copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        render_pkg = getattr(modules, get_render_func(dataset.base_model))(viewpoint_cam, gaussians, pipe, scene.background, iteration, dataset.render_mode)
        image, scaling = render_pkg["render"], render_pkg["scaling"]

        gt_image = viewpoint_cam.original_image.cuda()
        
        Ll1 = l1_loss(image, gt_image)
        ssim_loss = (1.0 - ssim(image, gt_image))
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss
       
        if opt.lambda_dreg > 0:
            if scaling.shape[0] > 0:
                scaling_reg = scaling.prod(dim=1).mean()
            else:
                scaling_reg = torch.tensor(0.0, device="cuda")
            loss += opt.lambda_dreg * scaling_reg
            

        if opt.lambda_normal > 0 and iteration > opt.normal_start_iter:
            # normal consistency loss
            normals = render_pkg["render_normals"].squeeze(0).permute((2, 0, 1))
            normals_from_depth = render_pkg["render_normals_from_depth"] * render_pkg["render_alphas"].squeeze(0).detach()
            if len(normals_from_depth.shape) == 4:
                normals_from_depth = normals_from_depth.squeeze(0)
            normals_from_depth = normals_from_depth.permute((2, 0, 1))
            normal_error = (1 - (normals * normals_from_depth).sum(dim=0))[None]
            loss += opt.lambda_normal * normal_error.mean()


        if opt.lambda_dist and iteration > opt.dist_start_iter:
            loss += opt.lambda_dist * render_pkg["render_distort"].mean()
            
        # if scale != 1.0:
        #     loss = loss * 0.01
            
        loss.backward()
        
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            eval_lpips = iteration == testing_iterations[-1]
            dump_images = iteration == testing_iterations[-1]
            training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, getattr(modules, get_render_func(dataset.base_model)), 
                            (pipe, scene.background, iteration, dataset.render_mode), wandb, logger, dataset.resolution_scales, eval_lpips=eval_lpips, dump_images=dump_images)
            if (iteration in saving_iterations):
                logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                size = scene.save(iteration)
                stat_dict['size'].append(size)
            
            # densification
            if iteration < opt.update_until and iteration > opt.start_stat:
                # add statis
                gaussians.training_statis(render_pkg, image.shape[2], image.shape[1])
                
                # densification
                if opt.densification and iteration > opt.update_from and iteration % opt.update_interval == 0:
                    gaussians.run_densify(iteration, opt)
            
            elif iteration == opt.update_until:
                gaussians.clean()
                    
            if args.test_only:
                return
            
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
            if (iteration in checkpoint_iterations):
                logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, wandb=None, logger=None, scales=[1.0, 2.0, 4.0, 8.0, 16.0],
                    eval_lpips=False, dump_images=False):
    
    if tb_writer:
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/iter_time', elapsed, iteration)


    if wandb is not None:
        wandb.log({"train_l1_loss":Ll1, 'train_total_loss':loss, })
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        scene.gaussians.eval()
        torch.cuda.empty_cache()
        
        # for lod_level in range(0, 5):
            
        for scale in scales:
            os.makedirs(os.path.join(lp.model_path, 'renders', f'scale={scale}'), exist_ok=True)
            validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras(scale)}, 
                                    {'name': 'train', 'cameras' : [scene.getTrainCameras(scale)[idx % len(scene.getTrainCameras(scale))] for idx in range(5, 30, 5)]})

            for config in validation_configs:
                if config['cameras'] and len(config['cameras']) > 0:
                    l1_test = 0.0
                    lpips_test = 0.0
                    ssim_test = 0.0
                    psnr_test = 0.0
                    num_gaussians = 0
                    fps = 0.0
                    
                    if wandb is not None:
                        gt_image_list = []
                        render_image_list = []
                        errormap_list = []

                    for idx, viewpoint in enumerate(config['cameras']):
                        pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs, measure_fps=True)
                        num_gaussians += pkg['visible_mask'].sum().item()
                        fps += pkg['fps']
                        image = torch.clamp(pkg["render"], 0.0, 1.0)
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                        
                        if dump_images and config['name'] == 'test':
                            np_img = (np.transpose(image.cpu().numpy(), (1, 2, 0)) * 255).astype(np.uint8)
                            img = Image.fromarray(np_img)
                            img.save(os.path.join(lp.model_path, 'renders', f'scale={scale}', f'{idx}.png'))
                        if tb_writer and (idx < 30):
                            tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                            tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/errormap".format(viewpoint.image_name), (gt_image[None]-image[None]).abs(), global_step=iteration)

                            if wandb:
                                render_image_list.append(image[None])
                                errormap_list.append((gt_image[None]-image[None]).abs())
                                
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                                if wandb:
                                    gt_image_list.append(gt_image[None])

                        l1_test += l1_loss(image, gt_image).mean().double()
                        psnr_test += psnr(image, gt_image).mean().double()
                        ssim_test += ssim(image, gt_image).mean().double()
                        if eval_lpips:
                            lpips_test += lpips_fn(image, gt_image, normalize=True).item()
                        

                    psnr_test /= len(config['cameras'])
                    l1_test /= len(config['cameras'])          
                    num_gaussians /= len(config['cameras'])
                    fps /= len(config['cameras'])
                    ssim_test /= len(config['cameras'])
                    if eval_lpips:
                        lpips_test /= len(config['cameras'])
                        
                    update_dict(config['name'], scale, psnr_test.item(), ssim_test.item(), lpips_test, fps, num_gaussians, scene.gaussians._anchor.shape[0])
                    
                    logger.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {} W {} FPS {}".format(iteration, config['name'], l1_test, psnr_test, num_gaussians, fps))
                    
                    if tb_writer:
                        tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                        tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    if wandb is not None:
                        wandb.log({f"{config['name']}_loss_viewpoint_l1_loss":l1_test, f"{config['name']}_PSNR":psnr_test})

            if tb_writer:
                tb_writer.add_scalar(f'{dataset_name}/'+'total_points', len(scene.gaussians.get_anchor), iteration)
            torch.cuda.empty_cache()

        scene.gaussians.train()


def render_set(base_model, model_path, name, iteration, views, gaussians, pipe, background, render_mode):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    t_list = []
    visible_count_list = []
    per_view_dict = {}
    modules = __import__('gaussian_renderer')
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        
        torch.cuda.synchronize();t_start = time.time()
        
        render_pkg = getattr(modules, get_render_func(base_model))(view, gaussians, pipe, background, iteration, render_mode)
        torch.cuda.synchronize();t_end = time.time()

        t_list.append(t_end - t_start)

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = render_pkg["visibility_filter"].sum()
        visible_count_list.append(visible_count)

        # gts
        gt = view.original_image[0:3, :, :]
        
        # error maps
        if gt.device != rendering.device:
            rendering = rendering.to(gt.device)
        errormap = (rendering - gt).abs()

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(errormap, os.path.join(error_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()
        
    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)
    
    return t_list, visible_count_list


def render_sets(dataset, opt, pipe, iteration, skip_train=False, skip_test=False, wandb=None, tb_writer=None, dataset_name=None, logger=None):
    with torch.no_grad():
        modules = __import__('scene.gs_model_'+dataset.base_model, fromlist=[''])
        model_config = dataset.model_config
        gaussians = getattr(modules, model_config['name'])(**model_config['kwargs'])
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, resolution_scales=dataset.resolution_scales)
        gaussians.eval()
        gaussians.set_coarse_interval(opt)
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)

        if not skip_train:
            t_train_list, visible_count = render_set(dataset.base_model, dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipe, scene.background, dataset.render_mode)
            train_fps = 1.0 / torch.tensor(t_train_list[5:]).mean()
            logger.info(f'Train FPS: \033[1;35m{train_fps.item():.5f}\033[0m')
            if tb_writer:
                tb_writer.add_scalar(f'{dataset_name}/train_FPS', test_fps.item(), 0)
            if wandb is not None:
                wandb.log({"train_fps":train_fps.item(), })

        if not skip_test:
            t_test_list, visible_count = render_set(dataset.base_model, dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipe, scene.background, dataset.render_mode)
            test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()
            logger.info(f'Test FPS: \033[1;35m{test_fps.item():.5f}\033[0m')
            if tb_writer:
                tb_writer.add_scalar(f'{dataset_name}/test_FPS', test_fps.item(), 0)
            if wandb is not None:
                wandb.log({"test_fps":test_fps, })
    
    return visible_count


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def evaluate(model_paths, eval_name, visible_count=None, wandb=None, tb_writer=None, dataset_name=None, logger=None):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    
    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / eval_name

    for method in os.listdir(test_dir):

        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}

        method_dir = test_dir / method
        gt_dir = method_dir/ "gt"
        renders_dir = method_dir / "renders"
        renders, gts, image_names = readImages(renders_dir, gt_dir)

        ssims = []
        psnrs = []
        lpipss = []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips_fn(renders[idx], gts[idx], normalize=True).detach())

        logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
        logger.info("  PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
        logger.info("  SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
        logger.info("  LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))
        logger.info("  GS_NUMS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(visible_count).float().mean(), ".5"))
        print("")

        if wandb is not None:
            wandb.log({"test_PSNR":torch.stack(psnrs).mean().item(), })
            wandb.log({"test_SSIM":torch.stack(ssims).mean().item(), })
            wandb.log({"test_LPIPS":torch.stack(lpipss).mean().item(), })
            wandb.log({"test_GS_NUMS":torch.stack(visible_count).float().mean().item(), })

        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/PSNR', torch.tensor(psnrs).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/SSIM', torch.tensor(ssims).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/LPIPS', torch.tensor(lpipss).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/GS_NUMS', torch.tensor(visible_count).float().mean().item(), 0)
        
        full_dict[scene_dir][method].update({
            "PSNR": torch.tensor(psnrs).mean().item(),
            "SSIM": torch.tensor(ssims).mean().item(),
            "LPIPS": torch.tensor(lpipss).mean().item(),
            "GS_NUMS": torch.tensor(visible_count).float().mean().item(),
            })

        per_view_dict[scene_dir][method].update({
            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
            "SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
            "GS_NUMS": {name: vc for vc, name in zip(torch.tensor(visible_count).tolist(), image_names)}
            })

    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)
    
    
def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO) 
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--config', type=str, help='train config file path')
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[-1])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[-1])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gpu", type=str, default = '-1')
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--ms_train", action='store_true', default=False)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--scene_name", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--source_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None,
                        help="Output directory; defaults to output/<dataset>/<scene>/<run>_base_model")
    
    args = parser.parse_args(sys.argv[1:])
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        lp, op, pp = parse_cfg(cfg)
        args.save_iterations.append(op.iterations)

    if args.run_name is None:
        args.run_name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    
    lp.scene_name = args.scene_name
    lp.dataset_name = args.dataset_name
    lp.source_path = args.source_path
    if args.model_path:
        lp.model_path = args.model_path
    else:
        repo = os.path.dirname(os.path.abspath(__file__))
        lp.model_path = os.path.join(repo, "output", lp.dataset_name or "", lp.scene_name or "",
                                     f"{args.run_name}_base_model")

    
    os.makedirs(lp.model_path, exist_ok=True)
    shutil.copy(args.config, os.path.join(lp.model_path, "config.yaml"))

    logger = get_logger(lp.model_path)
    


    if args.test_iterations[0] == -1:
        args.test_iterations = [i for i in range(1000, op.iterations + 1, 1000)]
        if args.test_only:
            args.test_iterations += [1]
    if len(args.test_iterations) == 0 or args.test_iterations[-1] != op.iterations:
        args.test_iterations.append(op.iterations)

    if args.save_iterations[0] == -1:
        args.save_iterations = [i for i in range(10000, op.iterations + 1, 10000)]
        args.save_iterations += [15_000]
        args.save_iterations += [25_000]
    if len(args.save_iterations) == 0 or args.save_iterations[-1] != op.iterations:
        args.save_iterations.append(op.iterations)

    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.system("echo $CUDA_VISIBLE_DEVICES")
        logger.info(f'using GPU {args.gpu}')

    exp_name = lp.scene_name if lp.dataset_name=="" else lp.dataset_name+"_"+lp.scene_name
    if args.use_wandb:
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project=f"Octree-GS",
            name=exp_name,
            # Track hyperparameters and run metadata
            settings=wandb.Settings(start_method="fork"),
            config=vars(args)
        )
    else:
        wandb = None
    
    logger.info("Optimizing " + lp.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # training
    
    ply_path=None
    lod_gradient_based = False

    training(lp, op, pp, exp_name, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb, logger, ply_path=ply_path)
    
    def save_to_json(filename, data):
        with open(filename, 'w') as json_file:
            json.dump(data, json_file, indent=4)
            
    
    save_to_json(os.path.join(lp.model_path, 'stat_dict.json'), stat_dict)


    def save_stats_to_csv(filename, data):
        def get_values_for_scale(key, scale):
            if isinstance(data.get(key, {}), dict):
                return data[key].get(scale, [])
            return data.get(key, [])  

        with open(filename, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['SCALE', 'PSNR', 'SSIM', 'LPIPS', 'FPS', 'ACTIVE_GAUSSIANS', 'NUM_GAUSSIANS', 'SIZE'])  # Intestazione

            for scale in lp.resolution_scales:
                # writer.writerow([
                #     scale,
                #     get_values_for_scale('psnr_train', scale)[-1] if get_values_for_scale('psnr_train', scale) else None,
                #     get_values_for_scale('ssim_train', scale)[-1] if get_values_for_scale('ssim_train', scale) else None,
                #     get_values_for_scale('lpips_train', scale)[-1] if get_values_for_scale('lpips_train', scale) else None,
                #     get_values_for_scale('fps_train', scale)[-1] if get_values_for_scale('fps_train', scale) else None,
                #     get_values_for_scale('active_gaussians_train', scale)[-1] if get_values_for_scale('active_gaussians_train', scale) else None,
                #     data.get('num_gaussians', [])[-1] if data.get('num_gaussians') else None,
                #     data.get('size', [])[-1] if data.get('size') else None
                # ])
                writer.writerow([
                    scale,
                    get_values_for_scale('psnr_test', scale)[-1] if get_values_for_scale('psnr_test', scale) else None,
                    get_values_for_scale('ssim_test', scale)[-1] if get_values_for_scale('ssim_test', scale) else None,
                    get_values_for_scale('lpips_test', scale)[-1] if get_values_for_scale('lpips_test', scale) else None,
                    get_values_for_scale('fps_test', scale)[-1] if get_values_for_scale('fps_test', scale) else None,
                    get_values_for_scale('active_gaussians_test', scale)[-1] if get_values_for_scale('active_gaussians_test', scale) else None,
                    data.get('num_gaussians', [])[-1] if data.get('num_gaussians') else None,
                    data.get('size', [])[-1] if data.get('size') else None
                ])
            
    save_stats_to_csv(os.path.join(lp.model_path, 'final_results2.csv'), stat_dict)

    # All done
    logger.info("\nTraining and Evaluation completed.")




