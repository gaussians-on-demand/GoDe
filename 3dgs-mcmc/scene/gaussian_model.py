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
import io
import pickle
import struct
from typing import OrderedDict
import zstandard as zstd
import lzma
import json
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, pcast_i16_to_f32
from utils.reloc_utils import compute_relocation_cuda


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.setup_functions()
        self.qa = False


    def set_qa(self):
        self.qa = True
        self.features_dc_qa = torch.ao.quantization.FakeQuantize(dtype=torch.qint8).cuda()
        self.features_rest_qa = torch.ao.quantization.FakeQuantize(dtype=torch.qint8).cuda()
        self.opacity_qa = FakeQuantizationHalf.apply
        self.scaling_qa = FakeQuantizationHalf.apply
        self.rotation_qa = FakeQuantizationHalf.apply
        self.xyz_qa = FakeQuantizationHalf.apply
        #self.xyz_qa = lambda x: x


    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        
    @property
    def get_scaling(self):
        if self.qa:
            return self.scaling_qa(self.scaling_activation(self._scaling))
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        if self.qa:
            return self.rotation_activation(self.rotation_qa(self._rotation))
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        if self.qa:
            return self.xyz_qa(self._xyz)
        return self._xyz
    
    @property
    def get_features(self):
        if self.qa:
            features_dc = self.features_dc_qa(self._features_dc)
            features_rest = self.features_rest_qa(self._features_rest)
        else:
            features_dc = self._features_dc
            features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    
    @property
    def get_opacity(self):
        if self.qa:
            return self.opacity_qa(self.opacity_activation(self._opacity))
        return self.opacity_activation(self._opacity)
    
    @property
    def num_primitives(self):
        return self._xyz.shape[0]
    
    @property
    def per_band_count(self):
        result = list()
        if self.variable_sh_bands:
            for tensor in self._features_rest:
                result.append(tensor.shape[0])
        return result
    
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2)*0.1)[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.5 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
    

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors


    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, reset_params=True):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        if reset_params:
            self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")


    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def replace_tensors_to_optimizer(self, inds=None):
        tensors_dict = {"xyz": self._xyz,
            "f_dc": self._features_dc,
            "f_rest": self._features_rest,
            "opacity": self._opacity,
            "scaling" : self._scaling,
            "rotation" : self._rotation}

        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            
            if inds is not None:
                stored_state["exp_avg"][inds] = 0
                stored_state["exp_avg_sq"][inds] = 0
            else:
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

            del self.optimizer.state[group['params'][0]]
            group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
            self.optimizer.state[group['params'][0]] = stored_state

            optimizable_tensors[group["name"]] = group["params"][0]

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"] 

        return optimizable_tensors

    
    def _update_params(self, idxs, ratio):
        new_opacity, new_scaling = compute_relocation_cuda(
            opacity_old=self.get_opacity[idxs, 0],
            scale_old=self.get_scaling[idxs],
            N=ratio[idxs, 0] + 1
        )
        new_opacity = torch.clamp(new_opacity.unsqueeze(-1), max=1.0 - torch.finfo(torch.float32).eps, min=0.005)
        new_opacity = self.inverse_opacity_activation(new_opacity)
        new_scaling = self.scaling_inverse_activation(new_scaling.reshape(-1, 3))

        return self._xyz[idxs], self._features_dc[idxs], self._features_rest[idxs], new_opacity, new_scaling, self._rotation[idxs]


    def _sample_alives(self, probs, num, alive_indices=None):
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
        sampled_idxs = torch.multinomial(probs, num, replacement=True)
        if alive_indices is not None:
            sampled_idxs = alive_indices[sampled_idxs]
        ratio = torch.bincount(sampled_idxs).unsqueeze(-1)
        return sampled_idxs, ratio
    

    def relocate_gs(self, dead_mask=None):

        if dead_mask.sum() == 0:
            return

        alive_mask = ~dead_mask 
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]

        if alive_indices.shape[0] <= 0:
            return

        # sample from alive ones based on opacity
        probs = (self.get_opacity[alive_indices, 0]) 
        reinit_idx, ratio = self._sample_alives(alive_indices=alive_indices, probs=probs, num=dead_indices.shape[0])

        (
            self._xyz[dead_indices], 
            self._features_dc[dead_indices],
            self._features_rest[dead_indices],
            self._opacity[dead_indices],
            self._scaling[dead_indices],
            self._rotation[dead_indices] 
        ) = self._update_params(reinit_idx, ratio=ratio)
        
        self._opacity[reinit_idx] = self._opacity[dead_indices]
        self._scaling[reinit_idx] = self._scaling[dead_indices]

        self.replace_tensors_to_optimizer(inds=reinit_idx) 
        

    def add_new_gs(self, cap_max):
        current_num_points = self._opacity.shape[0]
        target_num = min(cap_max, int(1.05 * current_num_points))
        num_gs = max(0, target_num - current_num_points)

        if num_gs <= 0:
            return 0

        probs = self.get_opacity.squeeze(-1) 
        add_idx, ratio = self._sample_alives(probs=probs, num=num_gs)

        (
            new_xyz, 
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation 
        ) = self._update_params(add_idx, ratio=ratio)

        self._opacity[add_idx] = new_opacity
        self._scaling[add_idx] = new_scaling

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, reset_params=False)
        self.replace_tensors_to_optimizer(inds=add_idx)

        return num_gs


    def gradient_prune(self, k, prune_type='gradient_all', largest=False):
        if prune_type is not None:
            if prune_type == 'random':
                indices = torch.multinomial(torch.ones((self._xyz.shape[0])), num_samples=k, replacement=False)
            elif prune_type == 'gradient_xyz':
                indices = torch.topk(self._xyz.grad.norm(dim=1), k=k, largest=largest)[1].squeeze()
            elif prune_type == 'gradient_w/o_feats':
                indices = torch.topk(self._xyz.grad.norm(dim=1) +
                                    self._opacity.grad.norm(dim=1) +
                                    self._scaling.grad.norm(dim=1) +
                                    self._rotation.grad.norm(dim=1), k=k, largest=largest)[1].squeeze()
            elif prune_type == 'gradient_all':
                indices = torch.topk(
                    self._opacity.grad.norm(dim=1) +
                    self._xyz.grad.norm(dim=1) +
                    self._scaling.grad.norm(dim=1) +
                    self._rotation.grad.norm(dim=1) +
                    self._features_rest.grad.norm(dim=-1).norm(dim=-1) +
                    self._features_dc.grad.norm(dim=-1).norm(dim=-1)
                    , k=k, largest=largest)[1].squeeze()
            
            elif prune_type == 'opacity':
                indices = torch.topk(self._opacity.squeeze(), k=k, largest=largest)[1].squeeze()
            else:
                print('Pruning type not supported...')
                exit()
        if largest:
            mask = torch.zeros([self._xyz.shape[0]]).bool()
            mask[indices] = True
        
        else:
            mask = torch.ones([self._xyz.shape[0]]).bool()
            mask[indices] = False
        
        return mask, None

            
    @torch.no_grad()
    def create_zstd_dictionary_from_quantized_gaussians(
        self,
        dict_sizes=(262_144,),
        save_path=None,
        block_size=1024,
        small_chunk_gauss=128,
        max_samples=80_000,
        target_corpus_mult=400,
        seed=1234,
    ):
        rng = np.random.default_rng(seed)
        self._sort_morton()
        fdc_q = torch.quantize_per_tensor(
            self._features_dc.detach(),
            self.features_dc_qa.scale, self.features_dc_qa.zero_point, self.features_dc_qa.dtype,
        ).int_repr().cpu().numpy()
        fr_q  = torch.quantize_per_tensor(
            self._features_rest.detach(),
            self.features_rest_qa.scale, self.features_rest_qa.zero_point, self.features_rest_qa.dtype,
        ).int_repr().cpu().numpy()
        xyz      = np.ascontiguousarray(self._xyz.half().detach().cpu().numpy())
        opacity  = np.ascontiguousarray(self._opacity.half().detach().cpu().numpy())
        scaling  = np.ascontiguousarray(self._scaling.half().detach().cpu().numpy())
        rotation = np.ascontiguousarray(self._rotation.half().detach().cpu().numpy())
        N = xyz.shape[0]
        total_blocks = max(1, N // block_size)
        block_ids = rng.permutation(total_blocks)
        samples_feat, samples_geom = [], []
        for i in block_ids:
            s = i * block_size
            e = min((i + 1) * block_size, N)
            for off in range(s, e, small_chunk_gauss):
                ss, ee = off, min(off + small_chunk_gauss, e)
                payload_feat = {"features_dc": fdc_q[ss:ee], "features_rest": fr_q[ss:ee]}
                payload_geom = {"xyz": xyz[ss:ee], "opacity": opacity[ss:ee], "scaling": scaling[ss:ee], "rotation": rotation[ss:ee]}
                samples_feat.append(pickle.dumps(payload_feat, protocol=pickle.HIGHEST_PROTOCOL))
                samples_geom.append(pickle.dumps(payload_geom, protocol=pickle.HIGHEST_PROTOCOL))
                if len(samples_feat) >= max_samples:
                    break
            if len(samples_feat) >= max_samples:
                break
        if len(samples_feat) < 100:
            raise ValueError("Training corpus too small.")
        val_payload_feat = pickle.dumps({"features_dc": fdc_q, "features_rest": fr_q}, protocol=pickle.HIGHEST_PROTOCOL)
        val_payload_geom = pickle.dumps({"xyz": xyz, "opacity": opacity, "scaling": scaling, "rotation": rotation}, protocol=pickle.HIGHEST_PROTOCOL)
        def _train_dict(ds, samples):
            try:
                return zstd.train_dictionary(ds, samples, k=2048, d=8, steps=8, split_point=0.75, shrink_dict=True, threads=0)
            except TypeError:
                return zstd.train_dictionary(ds, samples)

        def _make_cctx(zdict):
            try:
                cparams = zstd.ZstdCompressionParameters.from_level(
                    level=22,
                    enable_ldm=True,
                    window_log=27,
                    ldm_min_match=64,
                    ldm_hash_log=22,
                    ldm_bucket_size_log=3,
                    ldm_hash_rate_log=4,
                )
                return zstd.ZstdCompressor(
                    compression_params=cparams,
                    dict_data=zdict,
                    write_content_size=False,
                    threads=-1,
                )
            except Exception:
                return zstd.ZstdCompressor(level=22, dict_data=zdict, write_content_size=False, threads=-1)

        def _measure_size(zdict, payload):
            cctx = _make_cctx(zdict)
            return len(cctx.compress(payload))
        best_feat = (None, None)
        best_geom = (None, None)
        for ds in dict_sizes:
            zdict_f = _train_dict(ds, samples_feat)
            size_f = _measure_size(zdict_f, val_payload_feat)
            if best_feat[1] is None or size_f < best_feat[1]:
                best_feat = (zdict_f, size_f)
            zdict_g = _train_dict(ds, samples_geom)
            size_g = _measure_size(zdict_g, val_payload_geom)
            if best_geom[1] is None or size_g < best_geom[1]:
                best_geom = (zdict_g, size_g)
        zstd_dicts = {"features": best_feat[0], "geom": best_geom[0]}
        if save_path is not None:
            data = {"features": zstd_dicts["features"].as_bytes(), "geom": zstd_dicts["geom"].as_bytes()}
            with open(save_path, "wb") as f:
                f.write(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
        return zstd_dicts
    
    
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        
        
    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree        
        

    def save_zstd(self, path, zstd_dict=None, compression_level=22):
        if isinstance(path, str):
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fdc_q = torch.quantize_per_tensor(
            self._features_dc.detach(),
            self.features_dc_qa.scale, self.features_dc_qa.zero_point, self.features_dc_qa.dtype,
        ).int_repr().cpu().numpy()
        fr_q  = torch.quantize_per_tensor(
            self._features_rest.detach(),
            self.features_rest_qa.scale, self.features_rest_qa.zero_point, self.features_rest_qa.dtype,
        ).int_repr().cpu().numpy()
        payload_features = {
            "quantization": True,
            "features_dc": fdc_q,
            "features_dc_scale": self.features_dc_qa.scale.cpu().numpy(),
            "features_dc_zero_point": self.features_dc_qa.zero_point.cpu().numpy(),
            "features_rest": fr_q,
            "features_rest_scale": self.features_rest_qa.scale.cpu().numpy(),
            "features_rest_zero_point": self.features_rest_qa.zero_point.cpu().numpy(),
        }
        payload_geom = {
            "xyz":  np.ascontiguousarray(self._xyz.half().detach().cpu().numpy()),
            "opacity": np.ascontiguousarray(self._opacity.half().detach().cpu().numpy()),
            "scaling": np.ascontiguousarray(self._scaling.half().detach().cpu().numpy()),
            "rotation": np.ascontiguousarray(self._rotation.half().detach().cpu().numpy()),
        }
        ser_features = pickle.dumps(payload_features, protocol=pickle.HIGHEST_PROTOCOL)
        ser_geom = pickle.dumps(payload_geom, protocol=pickle.HIGHEST_PROTOCOL)
        if isinstance(zstd_dict, dict):
            zdict_feat = zstd_dict.get("features", None)
            zdict_geom = zstd_dict.get("geom", None)
            if isinstance(zdict_feat, (bytes, bytearray)):
                zdict_feat = zstd.ZstdCompressionDict(zdict_feat)
            if isinstance(zdict_geom, (bytes, bytearray)):
                zdict_geom = zstd.ZstdCompressionDict(zdict_geom)
        else:
            zdict_feat = zstd_dict
            zdict_geom = zstd_dict
        
        def _make_cctx(level, zdict):
            try:
                cparams = zstd.ZstdCompressionParameters.from_level(
                    level=level,
                    enable_ldm=True,
                    window_log=27,
                    ldm_min_match=64,
                    ldm_hash_log=22,
                    ldm_bucket_size_log=3,
                    ldm_hash_rate_log=4,
                )
                return zstd.ZstdCompressor(
                    compression_params=cparams,
                    dict_data=zdict,
                    write_content_size=False,
                    threads=-1,
                )
            except Exception:
                return zstd.ZstdCompressor(level=level, dict_data=zdict, write_content_size=False, threads=-1)

        cctx_feat = _make_cctx(compression_level, zdict_feat)
        cctx_geom = _make_cctx(compression_level, zdict_geom)
        comp_features = cctx_feat.compress(ser_features)
        comp_geom = cctx_geom.compress(ser_geom)
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(comp_features)))
            f.write(comp_features)
            f.write(struct.pack("<Q", len(comp_geom)))
            f.write(comp_geom)
        size_mb = (len(comp_features) + len(comp_geom) + 16) / (1024 * 1024)
        print(f"[ZSTD] Saved {path} ({size_mb:.2f} MB)")
        return size_mb


    def load_zstd(self, path, zstd_dict_path=None):
        import io, struct, pickle, numpy as np, torch, zstandard as zstd
        from torch import nn

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        zdict_feat = None
        zdict_geom = None
        if zstd_dict_path is not None:
            with open(zstd_dict_path, "rb") as f:
                blob = f.read()
            try:
                obj = pickle.loads(blob)
                if isinstance(obj, dict) and "features" in obj and "geom" in obj:
                    zdict_feat = zstd.ZstdCompressionDict(obj["features"])
                    zdict_geom = zstd.ZstdCompressionDict(obj["geom"])
                else:
                    zdict_feat = zstd.ZstdCompressionDict(blob)
                    zdict_geom = zdict_feat
            except Exception:
                zdict_feat = zstd.ZstdCompressionDict(blob)
                zdict_geom = zdict_feat

        with open(path, "rb") as f:
            data = f.read()
        if len(data) < 16:
            raise ValueError("Corrupted file")

        n1 = struct.unpack_from("<Q", data, 0)[0]
        off = 8
        if off + n1 + 8 > len(data):
            raise ValueError("Inconsistent header")
        comp1 = data[off:off+n1]
        off += n1
        n2 = struct.unpack_from("<Q", data, off)[0]
        off += 8
        if off + n2 > len(data):
            raise ValueError("Inconsistent header")
        comp2 = data[off:off+n2]

        dctx_feat = zstd.ZstdDecompressor(dict_data=zdict_feat) if zdict_feat is not None else zstd.ZstdDecompressor()
        dctx_geom = zstd.ZstdDecompressor(dict_data=zdict_geom) if zdict_geom is not None else zstd.ZstdDecompressor()

        def _stream_decomp(dctx, blob):
            bio = io.BytesIO(blob)
            with dctx.stream_reader(bio) as r:
                chunks = []
                while True:
                    c = r.read(131072)
                    if not c:
                        break
                    chunks.append(c)
            return b"".join(chunks)

        ser_features = _stream_decomp(dctx_feat, comp1)
        ser_geom = _stream_decomp(dctx_geom, comp2)

        combined_features = pickle.loads(ser_features)
        combined_geom = pickle.loads(ser_geom)

        fr_q  = torch.from_numpy(np.ascontiguousarray(combined_features["features_rest"])).to(torch.int32).to(device)
        fr_s  = torch.from_numpy(np.ascontiguousarray(combined_features["features_rest_scale"])).to(torch.float32).to(device)
        fr_zp = torch.from_numpy(np.ascontiguousarray(combined_features["features_rest_zero_point"])).to(torch.int32).to(device)
        if fr_s.ndim == 0: fr_s = fr_s.view(1)
        if fr_zp.ndim == 0: fr_zp = fr_zp.view(1)
        features_rest = (fr_q - fr_zp.to(fr_q.dtype)) * fr_s
        self._features_rest = nn.Parameter(features_rest.float(), requires_grad=True)
        if hasattr(self, "features_rest_qa"):
            self.features_rest_qa.scale = fr_s; self.features_rest_qa.zero_point = fr_zp

        fdc_q  = torch.from_numpy(np.ascontiguousarray(combined_features["features_dc"])).to(torch.int32).to(device)
        fdc_s  = torch.from_numpy(np.ascontiguousarray(combined_features["features_dc_scale"])).to(torch.float32).to(device)
        fdc_zp = torch.from_numpy(np.ascontiguousarray(combined_features["features_dc_zero_point"])).to(torch.int32).to(device)
        if fdc_s.ndim == 0: fdc_s = fdc_s.view(1)
        if fdc_zp.ndim == 0: fdc_zp = fdc_zp.view(1)
        features_dc = (fdc_q - fdc_zp.to(fdc_q.dtype)) * fdc_s
        self._features_dc = nn.Parameter(features_dc.float(), requires_grad=True)
        if hasattr(self, "features_dc_qa"):
            self.features_dc_qa.scale = fdc_s; self.features_dc_qa.zero_point = fdc_zp

        self._xyz      = torch.from_numpy(np.ascontiguousarray(combined_geom["xyz"])).to(torch.float32).to(device)
        self._rotation = torch.from_numpy(np.ascontiguousarray(combined_geom["rotation"])).to(torch.float32).to(device)
        self._scaling  = torch.from_numpy(np.ascontiguousarray(combined_geom["scaling"])).to(torch.float32).to(device)
        self._opacity  = torch.from_numpy(np.ascontiguousarray(combined_geom["opacity"])).to(torch.float32).to(device)
        return True


    def load_npz(self, path):
        
        with open(path, 'rb') as file:
            compressed_data = file.read()

        serialized_data = lzma.decompress(compressed_data)
        combined_data = pickle.loads(serialized_data)
        

        features_rest_q = torch.from_numpy(combined_data["features_rest"]).int().cuda()
        features_rest_scale = torch.from_numpy(combined_data["features_rest_scale"]).cuda()
        features_rest_zero_point = torch.from_numpy(combined_data["features_rest_zero_point"]).cuda()
        features_rest = (features_rest_q - features_rest_zero_point) * features_rest_scale
        self._features_rest = nn.Parameter(features_rest, requires_grad=True)
        self.features_rest_qa.scale = features_rest_scale
        self.features_rest_qa.zero_point = features_rest_zero_point
        self.features_rest_qa.activation_post_process.min_val = features_rest.min()
        self.features_rest_qa.activation_post_process.max_val = features_rest.max()

        features_dc_q = torch.from_numpy(combined_data["features_dc"]).int().cuda()
        features_dc_scale = torch.from_numpy(combined_data["features_dc_scale"]).cuda()
        features_dc_zero_point = torch.from_numpy(combined_data["features_dc_zero_point"]).cuda()
        features_dc = (features_dc_q - features_dc_zero_point) * features_dc_scale
        self._features_dc = nn.Parameter(features_dc, requires_grad=True)
        self.features_dc_qa.scale = features_dc_scale
        self.features_dc_qa.zero_point = features_dc_zero_point
        self.features_dc_qa.activation_post_process.min_val = features_dc.min()
        self.features_dc_qa.activation_post_process.max_val = features_dc.max()

        self._xyz = torch.from_numpy(combined_data['xyz']).cuda()
        self._rotation = torch.from_numpy(combined_data['rotation']).cuda()
        self._scaling = torch.from_numpy(combined_data['scaling']).cuda()
        self._opacity = torch.from_numpy(combined_data['opacity']).cuda()

        self.active_sh_degree = combined_data['active_degree']
    
    
    @torch.no_grad()
    def save_npz(self, path, sort_morton=True):
        
        if sort_morton:
            self._sort_morton()
        if isinstance(path, str):
            mkdir_p(os.path.dirname(os.path.abspath(path)))

        #dtype = torch.half if half_precision else torch.float32

        save_dict = dict()

        save_dict["quantization"] = self.qa

        # save position
        # if self.qa:
        #     save_dict["xyz"] = self.get_xyz.detach().half().cpu().numpy()
        # else:
        #     save_dict["xyz"] = self._xyz.detach().cpu().numpy()

        # save color features
        if self.qa:
            features_dc_q = torch.quantize_per_tensor(
                self._features_dc.detach(),
                self.features_dc_qa.scale,
                self.features_dc_qa.zero_point,
                self.features_dc_qa.dtype,
            ).int_repr()
            save_dict["features_dc"] = features_dc_q.cpu().numpy()
            save_dict["features_dc_scale"] = self.features_dc_qa.scale.cpu().numpy()
            save_dict[
                "features_dc_zero_point"
            ] = self.features_dc_qa.zero_point.cpu().numpy()

            features_rest_q = torch.quantize_per_tensor(
                self._features_rest.detach(),
                self.features_rest_qa.scale,
                self.features_rest_qa.zero_point,
                self.features_rest_qa.dtype,
            ).int_repr()
            save_dict["features_rest"] = features_rest_q.cpu().numpy()
            save_dict["features_rest_scale"] = self.features_rest_qa.scale.cpu().numpy()
            save_dict[
                "features_rest_zero_point"
            ] = self.features_rest_qa.zero_point.cpu().numpy()
            
            save_dict['xyz'] = self._xyz.detach().cpu().numpy()
            save_dict['opacity'] = self._opacity.half().detach().cpu().numpy()
            save_dict['scaling'] = self._scaling.half().detach().cpu().numpy()
            save_dict['rotation'] = self._rotation.half().detach().cpu().numpy()
            
        else:
            save_dict["features_dc"] = self._features_dc.detach().cpu().numpy()
            save_dict["features_rest"] = self._features_rest.detach().cpu().numpy()
            save_dict['xyz'] = self._xyz.cpu().numpy()
            save_dict['opacity'] = self._opacity.detach().cpu().numpy()
            save_dict['scaling'] = self._scaling.detach().cpu().numpy()
            save_dict['rotation'] = self._rotation.detach().cpu().numpy()
            
        save_dict['active_degree'] = self.active_sh_degree
            
        serialized_data = pickle.dumps(save_dict)

        compressed_data = lzma.compress(serialized_data)

        compressed_size_bytes = len(compressed_data)

        compressed_size_mb = compressed_size_bytes / (1024 * 1024)  # Convert bytes to MB

        print(f"Size of the compressed data: {compressed_size_mb:.2f} MB")
        
        with open(path, 'wb') as file:
            file.write(compressed_data)
            
        return compressed_size_mb
        
    
    def _sort_morton(self):
        with torch.no_grad():
            xyz_q = (
                (2**21 - 1)
                * (self._xyz - self._xyz.min(0).values)
                / (self._xyz.max(0).values - self._xyz.min(0).values)
            ).long()
            order = mortonEncode(xyz_q).sort().indices
            self._xyz = nn.Parameter(self._xyz[order], requires_grad=True)
            self._opacity = nn.Parameter(self._opacity[order], requires_grad=True)
            # self._scaling_factor = nn.Parameter(
            #     self._scaling_factor[order], requires_grad=True
            # )

            # if self.is_color_indexed:
            #     self._feature_indices = nn.Parameter(
            #         self._feature_indices[order], requires_grad=False
            #     )
            #else:
            self._features_rest = nn.Parameter(
                self._features_rest[order], requires_grad=True
            )
            self._features_dc = nn.Parameter(
                self._features_dc[order], requires_grad=True
            )

            # if self.is_gaussian_indexed:
            #     self._gaussian_indices = nn.Parameter(
            #         self._gaussian_indices[order], requires_grad=False
            #     )
            # else:
            self._scaling = nn.Parameter(self._scaling[order], requires_grad=True)
            self._rotation = nn.Parameter(self._rotation[order], requires_grad=True)
            return order
            
            
class FakeQuantizationHalf(torch.autograd.Function):
    """performs fake quantization for half precision"""

    @staticmethod
    def forward(_, x: torch.Tensor) -> torch.Tensor:
        return x.half().float()

    @staticmethod
    def backward(_, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output
    

def splitBy3(a):
    x = a & 0x1FFFFF  # we only look at the first 21 bits
    x = (x | x << 32) & 0x1F00000000FFFF
    x = (x | x << 16) & 0x1F0000FF0000FF
    x = (x | x << 8) & 0x100F00F00F00F00F
    x = (x | x << 4) & 0x10C30C30C30C30C3
    x = (x | x << 2) & 0x1249249249249249
    
    return x


def mortonEncode(pos: torch.Tensor) -> torch.Tensor:
    x, y, z = pos.unbind(-1)
    answer = torch.zeros(len(pos), dtype=torch.long, device=pos.device)
    answer |= splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2
    
    return answer