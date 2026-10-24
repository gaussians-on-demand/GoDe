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
import time
import math
import torch
import numpy as np
from torch import nn
from einops import repeat
from utils.system_utils import mkdir_p
from torch_scatter import scatter_max
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from scene.basic_model import BasicModel
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, knn
import pickle
import lzma

class GaussianLoDModel(BasicModel):

    def __init__(self, **model_kwargs):
        self.active_sh_degree = 0
        for key, value in model_kwargs.items():
            setattr(self, key, value)
        
        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._level = torch.empty(0)
        self._extra_level = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.qa = False

        self.opacity_accum = torch.empty(0)
        self.anchor_demon = torch.empty(0)
        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)

        self.optimizer = None
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._anchor,
            self._offset,
            self._level,
            self._extra_level,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.opacity_accum, 
            self.anchor_demon,
            self.offset_gradient_accum,
            self.offset_denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._anchor,
        self._offset,
        self._level,
        self._extra_level,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        self.opacity_accum, 
        self.anchor_demon,
        self.offset_gradient_accum,
        self.offset_denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_anchor(self):
        if self.qa:
            return self._anchor_qa(self._anchor)
        return self._anchor

    @property
    def get_level(self):
        return self._level
    
    @property
    def get_extra_level(self):
        if self.qa:
            return self.extra_level_qa(self._extra_level)
        return self._extra_level

    @property
    def get_offset(self):
        if self.qa:
            return self._offset_qa(self._offset)
        return self._offset

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
    def get_scaling(self):
        if self.qa:
            return self.scaling_qa(self.scaling_activation(self._scaling))
        else:
            return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        if self.qa:
            return self.rotation_activation(self.rotation_qa(self._rotation))
        return self.rotation_activation(self._rotation)


    def set_coarse_interval(self, opt):
        self.coarse_intervals = []
        num_level = self.levels - 1 - self.init_level
        if num_level > 0:
            q = 1/opt.coarse_factor
            a1 = opt.coarse_iter*(1-q)/(1-q**num_level)
            temp_interval = 0
            for i in range(num_level):
                interval = a1 * q ** i + temp_interval
                temp_interval = interval
                self.coarse_intervals.append(interval)

    def set_level(self, points, cameras, scales):
        all_dist = torch.tensor([]).cuda()
        self.cam_infos = torch.empty(0, 4).float().cuda()
        for scale in [1.0]:
            for cam in cameras[scale]:
                cam_center = cam.camera_center
                cam_info = torch.tensor([cam_center[0], cam_center[1], cam_center[2], scale]).float().cuda()
                self.cam_infos = torch.cat((self.cam_infos, cam_info.unsqueeze(dim=0)), dim=0)
                dist = torch.sqrt(torch.sum((points - cam_center)**2, dim=1))
                dist_max = torch.quantile(dist, self.dist_ratio)
                dist_min = torch.quantile(dist, 1 - self.dist_ratio)
                new_dist = torch.tensor([dist_min, dist_max]).float().cuda()
                new_dist = new_dist * scale
                all_dist = torch.cat((all_dist, new_dist), dim=0)
        dist_max = torch.quantile(all_dist, self.dist_ratio)
        dist_min = torch.quantile(all_dist, 1 - self.dist_ratio)
        self.standard_dist = dist_max
        if self.levels == -1:
            self.levels = torch.round(torch.log2(dist_max/dist_min)/math.log2(self.fork)).int().item() + 1
        if self.init_level == -1:
            self.init_level = int(self.levels/2)

    def octree_sample(self, points, colors):
        torch.cuda.synchronize(); t0 = time.time()
        self.positions = torch.empty(0, 3).float().cuda()
        self.colors = torch.empty(0, 3).float().cuda()
        self._level = torch.empty(0).int().cuda() 
        for cur_level in range(self.levels):
            cur_size = self.voxel_size/(float(self.fork) ** cur_level)
            new_candidates = torch.round((points - self.init_pos) / cur_size)
            new_candidates_unique, inverse_indices = torch.unique(new_candidates, return_inverse=True, dim=0)
            new_positions = new_candidates_unique * cur_size + self.init_pos
            new_positions += self.padding * cur_size
            new_levels = torch.ones(new_positions.shape[0], dtype=torch.int, device="cuda") * cur_level
            new_colors = scatter_max(colors, inverse_indices.unsqueeze(1).expand(-1, colors.size(1)), dim=0)[0]
            self.positions = torch.concat((self.positions, new_positions), dim=0)
            self.colors = torch.concat((self.colors, new_colors), dim=0)
            self._level = torch.concat((self._level, new_levels), dim=0)
        torch.cuda.synchronize(); t1 = time.time()
        time_diff = t1 - t0
        print(f"Building octree time: {int(time_diff // 60)} min {time_diff % 60} sec")

    def create_from_pcd(self, pcd, spatial_lr_scale, logger, *args):
        points = torch.tensor(pcd.points, dtype=torch.float, device="cuda")
        colors = torch.tensor(pcd.colors, dtype=torch.float, device="cuda")
        self.set_level(points, *args)
        self.spatial_lr_scale = spatial_lr_scale
        box_min = torch.min(points)*self.extend
        box_max = torch.max(points)*self.extend
        box_d = box_max - box_min
        print(box_d)
        if self.base_layer < 0:
            default_voxel_size = 0.02
            self.base_layer = torch.round(torch.log2(box_d/default_voxel_size)).int().item()-(self.levels//2)+1
        self.voxel_size = box_d/(float(self.fork) ** self.base_layer)
        self.init_pos = torch.tensor([box_min, box_min, box_min]).float().cuda()
        self.octree_sample(points, colors)

        if self.visible_threshold < 0:
            self.visible_threshold = 0.0
            self.positions, self._level, self.visible_threshold, _ = self.weed_out(self.positions, self._level)
        self.positions, self._level, _, weed_mask = self.weed_out(self.positions, self._level)
        self.colors = self.colors[weed_mask]

        logger.info(f'Branches of Tree: {self.fork}')
        logger.info(f'Base Layer of Tree: {self.base_layer}')
        logger.info(f'Visible Threshold: {self.visible_threshold}')
        logger.info(f'LOD Levels: {self.levels}')
        logger.info(f'Initial Levels: {self.init_level}')
        logger.info(f'Initial Voxel Number: {self.positions.shape[0]}')
        logger.info(f'Min Voxel Size: {self.voxel_size/(2.0 ** (self.levels - 1))}')
        logger.info(f'Max Voxel Size: {self.voxel_size}')

        fused_point_cloud, fused_color = self.positions, RGB2SH(self.colors)
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        dist2 = (knn(fused_point_cloud, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 6)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))        
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._level = self._level.unsqueeze(dim=1)
        self._extra_level = torch.zeros(self._anchor.shape[0], dtype=torch.float, device="cuda")
        self._anchor_mask = torch.ones(self._anchor.shape[0], dtype=torch.bool, device="cuda")

    def weed_out(self, gaussian_positions, gaussian_levels):
        visible_count = torch.zeros(gaussian_positions.shape[0], dtype=torch.int, device="cuda")
        for cam in self.cam_infos:
            cam_center, scale = cam[:3], cam[3]
            dist = torch.sqrt(torch.sum((gaussian_positions - cam_center)**2, dim=1)) * scale
            pred_level = torch.log2(self.standard_dist/dist)/math.log2(self.fork)   
            int_level = self.map_to_int_level(pred_level, self.levels - 1)
            visible_count += (gaussian_levels <= int_level).int()
        visible_count = visible_count/len(self.cam_infos)
        weed_mask = (visible_count > self.visible_threshold)
        mean_visible = torch.mean(visible_count)
        return gaussian_positions[weed_mask], gaussian_levels[weed_mask], mean_visible, weed_mask

    def set_anchor_mask(self, cam_center, iteration, resolution_scale, anchor, level, extra_level, disable_lod=False):
        
        if disable_lod:
            self._anchor_mask = torch.ones(anchor.shape[0], dtype=torch.bool, device="cuda")
        else:
            dist = torch.sqrt(torch.sum((anchor - cam_center)**2, dim=1)) * resolution_scale
            pred_level = torch.log2(self.standard_dist/dist)/math.log2(self.fork) + extra_level
            
            if self.progressive:
                coarse_index = np.searchsorted(self.coarse_intervals, iteration) + 1 + self.init_level
            else:
                coarse_index = self.levels

            int_level = self.map_to_int_level(pred_level, coarse_index - 1)
            self._anchor_mask = (level.squeeze(dim=1) <= int_level)    
        
        # if up_to_l is not None:
        #     self._anchor_mask = torch.logical_and((self._level.squeeze(dim=1) <= up_to_l), self._anchor_mask)


    def set_anchor_mask_perlevel(self, cam_center, resolution_scale, cur_level):
        dist = torch.sqrt(torch.sum((self.get_anchor - cam_center)**2, dim=1)) * resolution_scale
        pred_level = torch.log2(self.standard_dist/dist)/math.log2(self.fork) + self._extra_level
        int_level = self.map_to_int_level(pred_level, cur_level)
        self._anchor_mask = (self._level.squeeze(dim=1) <= int_level)


    def training_setup(self, training_args):
        
        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
        
        l = [
            {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
            {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},            
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "features_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "features_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)
        
        
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            self.oneupSHdegree()

    def construct_list_of_attributes(self):
        l = []
        l.append('x')
        l.append('y')
        l.append('z')
        l.append('level')
        l.append('extra_level')
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
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

    def save_ply(self, path, iteration):
        mkdir_p(os.path.dirname(path))

        if self.progressive:
            coarse_index = np.searchsorted(self.coarse_intervals, iteration) + 1 + self.init_level
        else:
            coarse_index = self.levels

        level_mask = (self._level <= coarse_index-1).squeeze(-1)
        anchor = self._anchor[level_mask].detach().cpu().numpy()
        levels = self._level[level_mask].detach().cpu().numpy()
        extra_levels = self._extra_level.unsqueeze(dim=1)[level_mask].detach().cpu().numpy()
        offsets = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous()[level_mask].cpu().numpy()
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous()[level_mask].cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous()[level_mask].cpu().numpy()
        opacities = self._opacity[level_mask].detach().cpu().numpy()
        scale = self._scaling[level_mask].detach().cpu().numpy()
        rotation = self._rotation[level_mask].detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, levels, extra_levels, offsets, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        plydata = PlyData([el], obj_info=[
            'standard_dist {:.6f}'.format(self.standard_dist),
            'levels {:.6f}'.format(self.levels),
            ])
        plydata.write(path)
        
        size = os.path.getsize(path)//2**20
        return size

    def load_ply(self, path):
        plydata = PlyData.read(path)
        infos = plydata.obj_info
        for info in infos:
            var_name = info.split(' ')[0]
            self.__dict__[var_name] = float(info.split(' ')[1])
    
        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        
        if 'level' in plydata.elements[0]:
            levels = np.asarray(plydata.elements[0]["level"])[... ,np.newaxis].astype(np.int16)
            extra_levels = np.asarray(plydata.elements[0]["extra_level"])[... ,np.newaxis].astype(np.float32)
            
        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))
        
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((anchor.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((anchor.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        
        if 'level' in plydata.elements[0]:
            self._level = torch.tensor(levels, dtype=torch.int, device="cuda")
            self._extra_level = torch.tensor(extra_levels, dtype=torch.float, device="cuda").squeeze(dim=1)
            self.levels = round(self.levels)
            if self.init_level == -1:
                self.init_level = int(self.levels/2)
                
        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree

            

    def prune_anchor(self, mask):
        valid_points_mask = ~mask
        
        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._features_dc = optimizable_tensors["features_dc"]
        self._features_rest = optimizable_tensors["features_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._level = self._level[valid_points_mask]
        self._extra_level = self._extra_level[valid_points_mask]


    def anchor_growing(self, iteration, grads, threshold, update_ratio, extra_ratio, extra_up, offset_mask, overlap):
        init_length = self.get_anchor.shape[0]
        grads[~offset_mask] = 0.0
        anchor_grads = torch.sum(grads.reshape(-1, self.n_offsets), dim=-1) / (torch.sum(offset_mask.reshape(-1, self.n_offsets), dim=-1) + 1e-6)
        for cur_level in range(self.levels):
            update_value = self.fork ** update_ratio
            level_mask = (self.get_level == cur_level).squeeze(dim=1)
            level_ds_mask = (self.get_level == cur_level + 1).squeeze(dim=1)
            if torch.sum(level_mask) == 0:
                continue
            cur_size = self.voxel_size / (float(self.fork) ** cur_level)
            ds_size = cur_size / self.fork
            # update threshold
            cur_threshold = threshold * (update_value ** cur_level)
            ds_threshold = cur_threshold * update_value
            extra_threshold = cur_threshold * extra_ratio
            # mask from grad threshold
            candidate_mask = (grads >= cur_threshold) & (grads < ds_threshold)
            candidate_ds_mask = (grads >= ds_threshold)
            candidate_extra_mask = (anchor_grads >= extra_threshold)

            length_inc = self.get_anchor.shape[0] - init_length
            if length_inc > 0 :
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc * self.n_offsets, dtype=torch.bool, device='cuda')], dim=0)
                candidate_ds_mask = torch.cat([candidate_ds_mask, torch.zeros(length_inc * self.n_offsets, dtype=torch.bool, device='cuda')], dim=0)
                candidate_extra_mask = torch.cat([candidate_extra_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)   
            
            repeated_mask = repeat(level_mask, 'n -> (n k)', k=self.n_offsets)
            candidate_mask = torch.logical_and(candidate_mask, repeated_mask)
            candidate_ds_mask = torch.logical_and(candidate_ds_mask, repeated_mask)
            candidate_extra_mask = torch.logical_and(candidate_extra_mask, level_mask)
            if ~self.progressive or iteration > self.coarse_intervals[-1]:
                self._extra_level += extra_up * candidate_extra_mask.float()    

            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:,:3].unsqueeze(dim=1)

            grid_coords = torch.round((self.get_anchor[level_mask]-self.init_pos)/cur_size - self.padding).int()
            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round((selected_xyz-self.init_pos)/cur_size - self.padding).int()
            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)
            if overlap:
                remove_duplicates = torch.ones(selected_grid_coords_unique.shape[0], dtype=torch.bool, device="cuda")
                candidate_anchor = selected_grid_coords_unique[remove_duplicates] * cur_size + self.init_pos + self.padding * cur_size
                new_level = torch.ones(candidate_anchor.shape[0], dtype=torch.int, device='cuda') * cur_level
                candidate_anchor, new_level, _, weed_mask = self.weed_out(candidate_anchor, new_level)
                remove_duplicates_clone = remove_duplicates.clone()
                remove_duplicates[remove_duplicates_clone] = weed_mask
            elif selected_grid_coords_unique.shape[0] > 0 and grid_coords.shape[0] > 0:
                remove_duplicates = self.get_remove_duplicates(grid_coords, selected_grid_coords_unique)
                remove_duplicates = ~remove_duplicates
                candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size + self.init_pos + self.padding * cur_size
                new_level = torch.ones(candidate_anchor.shape[0], dtype=torch.int, device='cuda') * cur_level
                candidate_anchor, new_level, _, weed_mask = self.weed_out(candidate_anchor, new_level)
                remove_duplicates_clone = remove_duplicates.clone()
                remove_duplicates[remove_duplicates_clone] = weed_mask
            else:
                candidate_anchor = torch.zeros([0, 3], dtype=torch.float, device='cuda')
                remove_duplicates = torch.zeros(selected_grid_coords_unique.shape[0], dtype=torch.bool, device='cuda')
                new_level = torch.zeros([0], dtype=torch.int, device='cuda')

            grid_coords_ds = torch.round((self.get_anchor[level_ds_mask]-self.init_pos)/ds_size-self.padding).int()
            selected_xyz_ds = all_xyz.view([-1, 3])[candidate_ds_mask]
            selected_grid_coords_ds = torch.round((selected_xyz_ds-self.init_pos)/ds_size-self.padding).int()
            selected_grid_coords_unique_ds, inverse_indices_ds = torch.unique(selected_grid_coords_ds, return_inverse=True, dim=0)
            if (~self.progressive or iteration > self.coarse_intervals[-1]) and cur_level < self.levels - 1:
                if overlap:
                    remove_duplicates_ds = torch.ones(selected_grid_coords_unique_ds.shape[0], dtype=torch.bool, device="cuda")
                    candidate_anchor_ds = selected_grid_coords_unique_ds[remove_duplicates_ds]*ds_size+self.init_pos+self.padding*ds_size
                    new_level_ds = torch.ones(candidate_anchor_ds.shape[0], dtype=torch.int, device='cuda') * (cur_level + 1)
                    candidate_anchor_ds, new_level_ds, _, weed_ds_mask = self.weed_out(candidate_anchor_ds, new_level_ds)
                    remove_duplicates_ds_clone = remove_duplicates_ds.clone()
                    remove_duplicates_ds[remove_duplicates_ds_clone] = weed_ds_mask
                elif selected_grid_coords_unique_ds.shape[0] > 0 and grid_coords_ds.shape[0] > 0:
                    remove_duplicates_ds = self.get_remove_duplicates(grid_coords_ds, selected_grid_coords_unique_ds)
                    remove_duplicates_ds = ~remove_duplicates_ds
                    candidate_anchor_ds = selected_grid_coords_unique_ds[remove_duplicates_ds]*ds_size+self.init_pos+self.padding*ds_size
                    new_level_ds = torch.ones(candidate_anchor_ds.shape[0], dtype=torch.int, device='cuda') * (cur_level + 1)
                    candidate_anchor_ds, new_level_ds, _, weed_ds_mask = self.weed_out(candidate_anchor_ds, new_level_ds)
                    remove_duplicates_ds_clone = remove_duplicates_ds.clone()
                    remove_duplicates_ds[remove_duplicates_ds_clone] = weed_ds_mask
                else:
                    candidate_anchor_ds = torch.zeros([0, 3], dtype=torch.float, device='cuda')
                    remove_duplicates_ds = torch.zeros(selected_grid_coords_unique_ds.shape[0], dtype=torch.bool, device='cuda')
                    new_level_ds = torch.zeros([0], dtype=torch.int, device='cuda')
            else:
                candidate_anchor_ds = torch.zeros([0, 3], dtype=torch.float, device='cuda')
                remove_duplicates_ds = torch.zeros(selected_grid_coords_unique_ds.shape[0], dtype=torch.bool, device='cuda')
                new_level_ds = torch.zeros([0], dtype=torch.int, device='cuda')

            if candidate_anchor.shape[0] + candidate_anchor_ds.shape[0] > 0:
                
                new_anchor = torch.cat([candidate_anchor, candidate_anchor_ds], dim=0)
                new_level = torch.cat([new_level, new_level_ds]).unsqueeze(dim=1).float().cuda()
                
                new_features = self.get_features.unsqueeze(dim=1).repeat([1, self.n_offsets, 1, 1]).view([-1, (self.max_sh_degree + 1) ** 2, 3])[candidate_mask]
                new_features = scatter_max(new_features, inverse_indices.unsqueeze(1).expand(-1, new_features.size(1)), dim=0)[0][remove_duplicates]
                new_features_ds = self.get_features.unsqueeze(dim=1).repeat([1, self.n_offsets, 1, 1]).view([-1, (self.max_sh_degree + 1) ** 2, 3])[candidate_ds_mask]
                new_features_ds = scatter_max(new_features_ds, inverse_indices_ds.unsqueeze(1).expand(-1, new_features_ds.size(1)), dim=0)[0][remove_duplicates_ds]
                new_features = torch.cat([new_features, new_features_ds], dim=0)
                new_features_dc = new_features[:, 0:1, :]
                new_features_rest = new_features[:, 1:, :]
                
                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))
                new_opacities_ds = inverse_sigmoid(0.1 * torch.ones((candidate_anchor_ds.shape[0], 1), dtype=torch.float, device="cuda"))                
                new_opacities = torch.cat([new_opacities, new_opacities_ds], dim=0)
                
                new_scaling = torch.ones_like(candidate_anchor).repeat([1,2]).float().cuda()*cur_size # *0.05
                new_scaling_ds = torch.ones_like(candidate_anchor_ds).repeat([1,2]).float().cuda()*ds_size # *0.05
                new_scaling = torch.cat([new_scaling, new_scaling_ds], dim=0)
                new_scaling = torch.log(new_scaling)
                
                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], dtype=torch.float, device='cuda')
                new_rotation_ds = torch.zeros([candidate_anchor_ds.shape[0], 4], dtype=torch.float, device='cuda')
                new_rotation = torch.cat([new_rotation, new_rotation_ds], dim=0)
                new_rotation[:,0] = 1.0

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()
                new_offsets_ds = torch.zeros_like(candidate_anchor_ds).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()
                new_offsets = torch.cat([new_offsets, new_offsets_ds], dim=0)

                new_extra_level = torch.zeros(candidate_anchor.shape[0], dtype=torch.float, device='cuda')
                new_extra_level_ds = torch.zeros(candidate_anchor_ds.shape[0], dtype=torch.float, device='cuda')
                new_extra_level = torch.cat([new_extra_level, new_extra_level_ds])
                
                d = {
                    "anchor": new_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "features_dc": new_features_dc,
                    "features_rest": new_features_rest,
                    "offset": new_offsets,
                    "opacity": new_opacities
                }   

                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()
                
                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._features_dc = optimizable_tensors["features_dc"]
                self._features_rest = optimizable_tensors["features_rest"]
                self._offset = optimizable_tensors["offset"]
                self._opacity = optimizable_tensors["opacity"]
                self._level = torch.cat([self._level, new_level], dim=0)
                self._extra_level = torch.cat([self._extra_level, new_extra_level], dim=0)
    
    def run_densify(self, iteration, opt):
        # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom # [N*k, 1]
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > opt.update_interval * opt.success_threshold * 0.5).squeeze(dim=1)
        
        self.anchor_growing(iteration, grads_norm, opt.densify_grad_threshold, opt.update_ratio, opt.extra_ratio, opt.extra_up, offset_mask, opt.overlap)
        
        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)
        
        # prune anchors
        prune_mask = (self.opacity_accum < opt.min_opacity*self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > opt.update_interval * opt.success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask) # [N] 
        
        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum
        
        # update opacity accum 
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
        
        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask)
            

    def generate_neural_gaussians(self, viewpoint_camera, xyz, scaling, opacity, rotation, color, visible_mask=None, ape_code=-1):
        ## view frustum filtering for acceleration    

        if visible_mask is None:
            visible_mask = torch.ones(self.get_anchor.shape[0], dtype=torch.bool, device = self.get_anchor.device)
        
        xyz = xyz[visible_mask]
        scaling = scaling[visible_mask]
        opacity = opacity[visible_mask]
        rotation = rotation[visible_mask]
        color = color[visible_mask]

        if self.dist2level=="progressive":
            prog = self._prog_ratio[visible_mask]
            transition_mask = self.transition_mask[visible_mask]
            prog[~transition_mask] = 1.0
            opacity = opacity * prog

        # offsets
        
        mask = torch.ones(xyz.shape[0], dtype=torch.bool, device="cuda")
        return xyz, color, opacity, scaling, rotation, self.active_sh_degree, mask
    
    
    # def gradient_prune(self, k, random=False, prune_type=None):
    #     if prune_type is not None:
    #         if prune_type == 'random':
    #             indices = torch.multinomial(torch.ones((self._xyz.shape[0])), num_samples=k, replacement=False)
    #         elif prune_type == 'gradient_xyz':
    #             indices = torch.topk(self._xyz.grad.norm(dim=1), k=k, largest=False)[1].squeeze()
    #         elif prune_type == 'gradient_w/o_feats':
    #             indices = torch.topk(self._xyz.grad.norm(dim=1) +
    #                                 self._opacity.grad.norm(dim=1) +
    #                                 self._scaling.grad.norm(dim=1) +
    #                                 self._rotation.grad.norm(dim=1), k=k, largest=False)[1].squeeze()
    #         elif prune_type == 'gradient_all':
    #             indices = torch.topk(
    #                 self._opacity.grad.norm(dim=1) +
    #                 self._xyz.grad.norm(dim=1) +
    #                 self._scaling.grad.norm(dim=1) +
    #                 self._rotation.grad.norm(dim=1) +
    #                 self._features_rest.grad.norm(dim=-1).norm(dim=-1) +
    #                 self._features_dc.grad.norm(dim=-1).norm(dim=-1)
    #                 , k=k, largest=False)[1].squeeze()
            
    #         elif prune_type == 'opacity':
    #             indices = torch.topk(self._opacity.squeeze(), k=k, largest=False)[1].squeeze()
    #         else:
    #             print('Pruning type not supported...')
    #             exit()
        
    #     else:
    #         indices = torch.topk(
    #                             self._opacity.grad.norm(dim=1) +
    #                             self._xyz.grad.norm(dim=1) +
    #                             self._scaling.grad.norm(dim=1) +
    #                             self._rotation.grad.norm(dim=1) +
    #                             self._features_rest.grad.norm(dim=-1).norm(dim=-1) +
    #                             self._features_dc.grad.norm(dim=-1).norm(dim=-1)
    #                             , k=k, largest=False)[1].squeeze()
            
    #     mask = torch.ones([self._xyz.shape[0]]).bool()
    #     mask[indices] = False
        
    #     return mask
    
    
    def compute_gradient_importance(self):
        """Return RMS-normalized gradient importance for each anchor."""
        parameters = {
            "opacity": self._opacity,
            "anchor": self._anchor,
            "offset": self._offset,
            "scaling": self._scaling,
            "rotation": self._rotation,
            "features_rest": self._features_rest,
            "features_dc": self._features_dc,
        }
        missing = [name for name, parameter in parameters.items() if parameter.grad is None]
        if missing:
            raise RuntimeError(
                "Missing gradients while computing pruning importance: "
                + ", ".join(missing)
            )

        contributions = {
            "opacity": self._opacity.grad.norm(dim=1),
            "anchor": self._anchor.grad.norm(dim=1),
            "offset": self._offset.grad.norm(dim=-1).norm(dim=-1),
            "scaling": self._scaling.grad.norm(dim=1),
            "rotation": self._rotation.grad.norm(dim=1),
            "features_rest": self._features_rest.grad.norm(dim=-1).norm(dim=-1),
            "features_dc": self._features_dc.grad.norm(dim=-1).norm(dim=-1),
        }
        normalized = []
        num_anchors = self._anchor.shape[0]
        for name, contribution in contributions.items():
            if contribution.ndim != 1 or contribution.shape[0] != num_anchors:
                raise RuntimeError(
                    f"Invalid gradient score shape for {name}: "
                    f"expected ({num_anchors},), got {tuple(contribution.shape)}"
                )
            if not torch.isfinite(contribution).all():
                raise RuntimeError(f"Non-finite gradient contribution for {name}.")

            epsilon = torch.finfo(contribution.dtype).eps
            rms = contribution.square().mean().sqrt().clamp_min(epsilon)
            normalized.append(contribution / rms)

        score = torch.stack(normalized, dim=0).sum(dim=0).detach()
        if not torch.isfinite(score).all():
            raise RuntimeError("Non-finite normalized gradient importance score.")
        return score

    def gradient_prune(self, k, u_mask=None, prune_type=None):
        if prune_type is not None:
            if prune_type == 'random':
                indices = torch.multinomial(torch.ones((self._xyz.shape[0])), num_samples=k, replacement=False)
            elif prune_type == 'gradient_xyz':
                indices = torch.topk(self._xyz.grad.norm(dim=1), k=k, largest=False)[1].squeeze()
            elif prune_type == 'gradient_w/o_feats':
                indices = torch.topk(self._xyz.grad.norm(dim=1) +
                                    self._opacity.grad.norm(dim=1) +
                                    self._scaling.grad.norm(dim=1) +
                                    self._rotation.grad.norm(dim=1), k=k, largest=False)[1].squeeze()
            elif prune_type == 'gradient_all':
                indices = torch.topk(
                    self._opacity.grad.norm(dim=1) +
                    self._xyz.grad.norm(dim=1) +
                    self._scaling.grad.norm(dim=1) +
                    self._rotation.grad.norm(dim=1) +
                    self._features_rest.grad.norm(dim=-1).norm(dim=-1) +
                    self._features_dc.grad.norm(dim=-1).norm(dim=-1)
                    , k=k, largest=False)[1].squeeze()
            
            elif prune_type == 'opacity':
                indices = torch.topk(self._opacity.squeeze(), k=k, largest=False)[1].squeeze()
            else:
                print('Pruning type not supported...')
                exit()
        
        else:
            
            grad_norms = self.compute_gradient_importance()
            if u_mask is None:
                u_mask = torch.ones(self._anchor.shape[0], dtype=torch.bool, device=self._anchor.device)

            masked_grad_norms = grad_norms.clone()
            masked_grad_norms[~u_mask] = float('inf')
            indices = torch.topk(masked_grad_norms, k=k, largest=False)[1]
            
        mask = torch.clone(u_mask).detach()
        mask[indices] = False
        
        return mask
    

    def gradient_prune_ordered(self, k, u_mask=None, prune_type=None):
        if prune_type is not None:
            if prune_type == 'random':
                indices = torch.multinomial(torch.ones((self._xyz.shape[0])), num_samples=k, replacement=False)
            elif prune_type == 'gradient_xyz':
                indices = torch.topk(self._xyz.grad.norm(dim=1), k=k, largest=False)[1].squeeze()
            elif prune_type == 'gradient_w/o_feats':
                indices = torch.topk(self._xyz.grad.norm(dim=1) +
                                    self._opacity.grad.norm(dim=1) +
                                    self._scaling.grad.norm(dim=1) +
                                    self._rotation.grad.norm(dim=1), k=k, largest=False)[1].squeeze()
            elif prune_type == 'gradient_all':
                indices = torch.topk(
                    self._opacity.grad.norm(dim=1) +
                    self._xyz.grad.norm(dim=1) +
                    self._scaling.grad.norm(dim=1) +
                    self._rotation.grad.norm(dim=1) +
                    self._features_rest.grad.norm(dim=-1).norm(dim=-1) +
                    self._features_dc.grad.norm(dim=-1).norm(dim=-1)
                    , k=k, largest=False)[1].squeeze()
            
            elif prune_type == 'opacity':
                indices = torch.topk(self._opacity.squeeze(), k=k, largest=False)[1].squeeze()
            else:
                print('Pruning type not supported...')
                exit()
        
        else:
            
            grad_norms = self.compute_gradient_importance()
            if u_mask is None:
                u_mask = torch.ones(self._anchor.shape[0], dtype=torch.bool, device=self._anchor.device)

            masked_grad_norms = grad_norms.clone()
            masked_grad_norms[~u_mask] = float('inf')
            indices = torch.topk(masked_grad_norms, k=k, largest=False, sorted=True)[1]
        
        return indices
    
    
    def set_qa(self, quantize_pos=False):
        self.qa = True
        self.quantize_pos = quantize_pos
        self.features_dc_qa = torch.ao.quantization.FakeQuantize(dtype=torch.qint8).cuda()
        self.features_rest_qa = torch.ao.quantization.FakeQuantize(dtype=torch.qint8).cuda()
        self.extra_level_qa = torch.ao.quantization.FakeQuantize(dtype=torch.qint8).cuda()
        self.opacity_qa = FakeQuantizationHalf.apply
        self.scaling_qa = FakeQuantizationHalf.apply
        self.rotation_qa = FakeQuantizationHalf.apply
        if quantize_pos:
            self._anchor_qa = FakeQuantizationHalf.apply
            self._offset_qa = FakeQuantizationHalf.apply
        else:
            self._anchor_qa = lambda x: x
            self._offset_qa = lambda x: x
        
    
    def build_lod_representation(self, masks):

        disjointed_masks = []
        for i in range(len(masks) - 1):
            xor_result = torch.logical_xor(masks[i], masks[i + 1])
            disjointed_masks.append(xor_result)
            
            
        self.levels = len(disjointed_masks)
        self.init_level = self.levels

        self._level = torch.ones((self._anchor.shape[0], 1), dtype=torch.int8).cuda() * -1
        self._extra_level = torch.zeros(self._anchor.shape[0], dtype=torch.float32).cuda()
        disjointed_masks.reverse()
        
        for i, mask in enumerate(disjointed_masks):
            self._level[mask] = i
            
        mask = (self._level != -1).squeeze()
        
        # masks.reverse()
        self._opacity = self._opacity[mask].clone().detach().requires_grad_(True)
        self._anchor = self._anchor[mask].clone().detach().requires_grad_(True)
        self._offset = self._offset[mask].clone().detach().requires_grad_(True)
        self._scaling = self._scaling[mask].clone().detach().requires_grad_(True)
        self._rotation = self._rotation[mask].clone().detach().requires_grad_(True)
        self._features_rest = self._features_rest[mask].clone().detach().requires_grad_(True)
        self._features_dc = self._features_dc[mask].clone().detach().requires_grad_(True)
        self._level = self._level[mask]
        self._extra_level = self._extra_level[mask]
    
    
    def _sort_morton(self):
        with torch.no_grad():
            xyz =  self.get_anchor + self.get_offset.view([-1, 3]) * self.get_scaling[:,:3]
            xyz_q = (
                (2**21 - 1)
                * (xyz - xyz.min(0).values)
                / (xyz.max(0).values - xyz.min(0).values)
            ).long()
            order = mortonEncode(xyz_q).sort().indices
            self._anchor = nn.Parameter(self._anchor[order], requires_grad=True)
            self._offset = nn.Parameter(self._offset[order], requires_grad=True)
            self._opacity = nn.Parameter(self._opacity[order], requires_grad=True)
            self._level = self._level[order]
            self._extra_level = self._extra_level[order]

            self._features_rest = nn.Parameter(
                self._features_rest[order], requires_grad=True
            )
            self._features_dc = nn.Parameter(
                self._features_dc[order], requires_grad=True
            )

            self._scaling = nn.Parameter(self._scaling[order], requires_grad=True)
            self._rotation = nn.Parameter(self._rotation[order], requires_grad=True)
            
        
    def compress_and_save(self, path, sort_morton=True):
        
        if sort_morton:
            self._sort_morton()
        if isinstance(path, str):
            mkdir_p(os.path.dirname(os.path.abspath(path)))
        
        save_dict = dict()
        save_dict["qa"] = self.qa
        

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
        
        
        extra_level_q = torch.quantize_per_tensor(
            self._extra_level,
            self.extra_level_qa.scale,
            self.extra_level_qa.zero_point,
            self.extra_level_qa.dtype,
        ).int_repr()
        save_dict["extra_level"] = extra_level_q.cpu().numpy()
        save_dict["extra_level_scale"] = self.extra_level_qa.scale.cpu().numpy()
        save_dict[
            "extra_level_zero_point"
        ] = self.extra_level_qa.zero_point.cpu().numpy()
        
        if self.quantize_pos:
            save_dict['offset'] = self._offset.half().detach().cpu().numpy()
            save_dict['anchor'] = self._anchor.half().detach().cpu().numpy()
        else:
            save_dict['offset'] = self._offset.detach().cpu().numpy()
            save_dict['anchor'] = self._anchor.detach().cpu().numpy()
        save_dict['opacity'] = self._opacity.half().detach().cpu().numpy()
        save_dict['scaling'] = self._scaling.half().detach().cpu().numpy()
        save_dict['rotation'] = self._rotation.half().detach().cpu().numpy()
        save_dict['level'] = self._level.to(torch.int8).detach().cpu().numpy()
        save_dict['active_degree'] = self.active_sh_degree

        serialized_data = pickle.dumps(save_dict)
        compressed_data = lzma.compress(serialized_data)
        compressed_size_bytes = len(compressed_data)
        compressed_size_mb = compressed_size_bytes / (1024 * 1024)  # Convert bytes to MB
        print(f"Size of the compressed data: {compressed_size_mb:.2f} MB")
        
        with open(path, 'wb') as file:
            file.write(compressed_data)
            
        return compressed_size_mb
    
    
    def load_compressed_data(self, path):
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
        
        extra_level_q = torch.from_numpy(combined_data["extra_level"]).int().cuda()
        extra_level_scale = torch.from_numpy(combined_data["extra_level_scale"]).cuda()
        extra_level_zero_point = torch.from_numpy(combined_data["extra_level_zero_point"]).cuda()
        extra_level = (extra_level_q - extra_level_zero_point) * extra_level_scale
        self._extra_level = nn.Parameter(extra_level, requires_grad=True)
        self.extra_level_qa.scale = extra_level_scale
        self.extra_level_qa.zero_point = extra_level_zero_point
        self.extra_level_qa.activation_post_process.min_val = extra_level.min()
        self.extra_level_qa.activation_post_process.max_val = extra_level.max()

        self._anchor = torch.from_numpy(combined_data['anchor']).cuda()
        self._offset = torch.from_numpy(combined_data['offset']).cuda()
        self._rotation = torch.from_numpy(combined_data['rotation']).cuda()
        self._scaling = torch.from_numpy(combined_data['scaling']).cuda()
        self._opacity = torch.from_numpy(combined_data['opacity']).cuda()
        self._level = torch.from_numpy(combined_data['level']).cuda()
        self._anchor_mask = torch.ones(self._anchor.shape[0], dtype=torch.bool, device="cuda")

        self.active_sh_degree = combined_data['active_degree']
        self._anchor_mask = torch.ones(self._anchor.shape[0], dtype=torch.bool, device="cuda")
        self.qa = combined_data['qa']


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