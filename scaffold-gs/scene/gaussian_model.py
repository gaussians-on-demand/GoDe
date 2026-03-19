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

import pickle
import torch
from functools import reduce
import numpy as np
import torch.ao.quantization
from torch_scatter import scatter_max
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.embedding import Embedding
import lzma
import zstandard as zstd


class FakeQuantizationHalf(torch.autograd.Function):
    """performs fake quantization for half precision"""

    @staticmethod
    def forward(_, x: torch.Tensor) -> torch.Tensor:
        return x.half().float()

    @staticmethod
    def backward(_, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output
    
    
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


    def __init__(self, 
                 feat_dim: int=32, 
                 n_offsets: int=5, 
                 voxel_size: float=0.01,
                 update_depth: int=3, 
                 update_init_factor: int=100,
                 update_hierachy_factor: int=4,
                 use_feat_bank : bool = False,
                 appearance_dim : int = 32,
                 ratio : int = 1,
                 add_opacity_dist : bool = False,
                 add_cov_dist : bool = False,
                 add_color_dist : bool = False,
                 quantization=False
                 ):

        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.voxel_size = voxel_size
        self.update_depth = update_depth
        self.update_init_factor = update_init_factor
        self.update_hierachy_factor = update_hierachy_factor
        self.use_feat_bank = use_feat_bank
        self.quantization = quantization

        self.appearance_dim = appearance_dim
        self.embedding_appearance = None
        self.ratio = ratio
        self.add_opacity_dist = add_opacity_dist
        self.add_cov_dist = add_cov_dist
        self.add_color_dist = add_color_dist

        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._anchor_feat = torch.empty(0)
        
        self.opacity_accum = torch.empty(0)

        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        
        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)

        self.anchor_demon = torch.empty(0)
                
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        
        if self.quantization:
            self._anchor_feat_qa = torch.ao.quantization.FakeQuantize(dtype=torch.qint8).cuda()
            self._anchor_qa = FakeQuantizationHalf.apply
            self._scaling_qa = FakeQuantizationHalf.apply
            self._offset_qa = FakeQuantizationHalf.apply
        
        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(3+1, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3),
                nn.Softmax(dim=1)
            ).cuda()

        self.opacity_dist_dim = 1 if self.add_opacity_dist else 0
        self.mlp_opacity = nn.Sequential(
            nn.Linear(feat_dim+3+self.opacity_dist_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, n_offsets),
            nn.Tanh()
        ).cuda()

        self.add_cov_dist = add_cov_dist
        self.cov_dist_dim = 1 if self.add_cov_dist else 0
        self.mlp_cov = nn.Sequential(
            nn.Linear(feat_dim+3+self.cov_dist_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 7*self.n_offsets),
        ).cuda()

        self.color_dist_dim = 1 if self.add_color_dist else 0
        self.mlp_color = nn.Sequential(
            nn.Linear(feat_dim+3+self.color_dist_dim+self.appearance_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3*self.n_offsets),
            nn.Sigmoid()
        ).cuda()


    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        if self.appearance_dim > 0:
            self.embedding_appearance.eval()
        if self.use_feat_bank:
            self.mlp_feature_bank.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        if self.appearance_dim > 0:
            self.embedding_appearance.train()
        if self.use_feat_bank:                   
            self.mlp_feature_bank.train()

    def capture(self):
        return (
            self._anchor,
            self._anchor_feat,
            self._offset,
            #self._local,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            #self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (
        self._anchor, 
        self._anchor_feat,
        self._offset,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)

        self.optimizer.load_state_dict(opt_dict)


    def set_appearance(self, num_cameras):
        if self.appearance_dim > 0:
            self.embedding_appearance = Embedding(num_cameras, self.appearance_dim).cuda()
            
    @property
    def get_anchor_feat(self):
        if self.quantization:
            return self._anchor_feat_qa(self._anchor_feat)
        return self._anchor_feat
    
    @property
    def get_offset(self):
        if self.quantization:
            return self._offset_qa(self._offset)
        return self._offset

    @property
    def get_appearance(self):
        return self.embedding_appearance

    @property
    def get_scaling(self):
        if self.quantization:
            return self._scaling_qa(1.0*self.scaling_activation(self._scaling))
        
        return 1.0*self.scaling_activation(self._scaling)
    
    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank
    
    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity
    
    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_anchor(self):
        if self.quantization:
            return self._anchor_qa(self._anchor)
        return self._anchor
    
    @property
    def set_anchor(self, new_anchor):
        assert self._anchor.shape == new_anchor.shape
        del self._anchor
        torch.cuda.empty_cache()
        self._anchor = new_anchor
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    def voxelize_sample(self, data=None, voxel_size=0.01):
        np.random.shuffle(data)
        data = np.unique(np.round(data/voxel_size), axis=0)*voxel_size
        
        return data

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        points = pcd.points[::self.ratio]

        if self.voxel_size <= 0:
            init_points = torch.tensor(points).float().cuda()
            init_dist = distCUDA2(init_points).float().cuda()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            del init_points
            torch.cuda.empty_cache()

        print(f'Initial voxel_size: {self.voxel_size}')
        
        
        points = self.voxelize_sample(points, voxel_size=self.voxel_size)
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()
        
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 6)
        
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        
        
        if self.use_feat_bank:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                
                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"},
            ]
        elif self.appearance_dim > 0:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"},
            ]
        else:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
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
        
        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)
        
        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)
        
        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                        lr_final=training_args.mlp_featurebank_lr_final,
                                                        lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                        max_steps=training_args.mlp_featurebank_lr_max_steps)
        if self.appearance_dim > 0:
            self.appearance_scheduler_args = get_expon_lr_func(lr_init=training_args.appearance_lr_init,
                                                        lr_final=training_args.appearance_lr_final,
                                                        lr_delay_mult=training_args.appearance_lr_delay_mult,
                                                        max_steps=training_args.appearance_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.appearance_dim > 0 and param_group["name"] == "embedding_appearance":
                lr = self.appearance_scheduler_args(iteration)
                param_group['lr'] = lr
            
            
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        anchor = self._anchor.detach().cpu().numpy()
        normals = np.zeros_like(anchor)
        anchor_feat = self._anchor_feat.detach().cpu().numpy()
        offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, normals, offset, anchor_feat, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply_sparse_gaussian(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        
        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))
        
        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))

        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))


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


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name']:
                continue
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


    # statis grad information to guide liftting. 
    def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask):
        # update opacity stats
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity<0] = 0
        
        temp_opacity = temp_opacity.view([-1, self.n_offsets])
        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)
        
        # update anchor visiting statis
        self.anchor_demon[anchor_visible_mask] += 1

        # update neural gaussian statis
        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter
        
        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1

        

        
    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name']:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            
            
        return optimizable_tensors

    def prune_anchor(self,mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

    
    def anchor_growing(self, grads, threshold, offset_mask):
        ## 
        init_length = self.get_anchor.shape[0]*self.n_offsets
        for i in range(self.update_depth):
            # update threshold
            cur_threshold = threshold*((self.update_hierachy_factor//2)**i)
            # mask from grad threshold
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)
            
            # random pick
            rand_mask = torch.rand_like(candidate_mask.float())>(0.5**(i+1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)
            
            length_inc = self.get_anchor.shape[0]*self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)

            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:,:3].unsqueeze(dim=1)
            
            # assert self.update_init_factor // (self.update_hierachy_factor**i) > 0
            # size_factor = min(self.update_init_factor // (self.update_hierachy_factor**i), 1)
            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            cur_size = self.voxel_size*size_factor
            
            grid_coords = torch.round(self.get_anchor / cur_size).int()

            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)


            ## split data for reducing peak memory calling
            use_chunk = True
            if use_chunk:
                chunk_size = 4096
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                remove_duplicates_list = []
                for i in range(max_iters):
                    cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                    remove_duplicates_list.append(cur_remove_duplicates)
                
                remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else:
                remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)

            remove_duplicates = ~remove_duplicates
            candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size

            
            if candidate_anchor.shape[0] > 0:
                new_scaling = torch.ones_like(candidate_anchor).repeat([1,2]).float().cuda()*cur_size # *0.05
                new_scaling = torch.log(new_scaling)
                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:,0] = 1.0

                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))

                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]

                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()

                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "opacity": new_opacities,
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
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._opacity = optimizable_tensors["opacity"]
                


    def adjust_anchor(self, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005, grow=True, prune=True):
        # # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom # [N*k, 1]
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval*success_threshold*0.5).squeeze(dim=1)
        
        if grow:
            self.anchor_growing(grads_norm, grad_threshold, offset_mask)
        
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
        if prune:
            # # prune anchors
            prune_mask = (self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1)
            anchors_mask = (self.anchor_demon > check_interval*success_threshold).squeeze(dim=1) # [N, 1]
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
        
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")
        print('Number of anchors:', self._anchor.shape[0])

    def save_mlp_checkpoints(self, path, mode = 'unite'):#split or unite
        mkdir_p(os.path.dirname(path))
        if mode == 'split':
            self.mlp_opacity.eval()
            opacity_mlp = torch.jit.trace(self.mlp_opacity, (torch.rand(1, self.feat_dim+3+self.opacity_dist_dim).cuda()))
            opacity_mlp.save(os.path.join(path, 'opacity_mlp.pt'))
            self.mlp_opacity.train()

            self.mlp_cov.eval()
            cov_mlp = torch.jit.trace(self.mlp_cov, (torch.rand(1, self.feat_dim+3+self.cov_dist_dim).cuda()))
            cov_mlp.save(os.path.join(path, 'cov_mlp.pt'))
            self.mlp_cov.train()

            self.mlp_color.eval()
            color_mlp = torch.jit.trace(self.mlp_color, (torch.rand(1, self.feat_dim+3+self.color_dist_dim+self.appearance_dim).cuda()))
            color_mlp.save(os.path.join(path, 'color_mlp.pt'))
            self.mlp_color.train()

            if self.use_feat_bank:
                self.mlp_feature_bank.eval()
                feature_bank_mlp = torch.jit.trace(self.mlp_feature_bank, (torch.rand(1, 3+1).cuda()))
                feature_bank_mlp.save(os.path.join(path, 'feature_bank_mlp.pt'))
                self.mlp_feature_bank.train()

            if self.appearance_dim:
                self.embedding_appearance.eval()
                emd = torch.jit.trace(self.embedding_appearance, (torch.zeros((1,), dtype=torch.long).cuda()))
                emd.save(os.path.join(path, 'embedding_appearance.pt'))
                self.embedding_appearance.train()

        elif mode == 'unite':
            if self.use_feat_bank:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    'feature_bank_mlp': self.mlp_feature_bank.state_dict(),
                    'appearance': self.embedding_appearance.state_dict()
                    }, os.path.join(path, 'checkpoints.pth'))
            elif self.appearance_dim > 0:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    'appearance': self.embedding_appearance.state_dict()
                    }, os.path.join(path, 'checkpoints.pth'))
            else:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    }, os.path.join(path, 'checkpoints.pth'))
        else:
            raise NotImplementedError


    def load_mlp_checkpoints(self, path, mode = 'unite'):#split or unite
        if mode == 'split':
            self.mlp_opacity = torch.jit.load(os.path.join(path, 'opacity_mlp.pt')).cuda()
            self.mlp_cov = torch.jit.load(os.path.join(path, 'cov_mlp.pt')).cuda()
            self.mlp_color = torch.jit.load(os.path.join(path, 'color_mlp.pt')).cuda()
            if self.use_feat_bank:
                self.mlp_feature_bank = torch.jit.load(os.path.join(path, 'feature_bank_mlp.pt')).cuda()
            if self.appearance_dim > 0:
                self.embedding_appearance = torch.jit.load(os.path.join(path, 'embedding_appearance.pt')).cuda()
        elif mode == 'unite':
            checkpoint = torch.load(os.path.join(path, 'checkpoints.pth'))
            self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
            self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
            self.mlp_color.load_state_dict(checkpoint['color_mlp'])
            if self.use_feat_bank:
                self.mlp_feature_bank.load_state_dict(checkpoint['feature_bank_mlp'])
            if self.appearance_dim > 0:
                self.embedding_appearance.load_state_dict(checkpoint['appearance'])
        else:
            raise NotImplementedError


    def gradient_prune(self, k, random=False):
        if random:
            indices = torch.multinomial(torch.ones((self._xyz.shape[0])), num_samples=k, replacement=False)
        else:
            # indices = torch.topk(
            #                     self._offset.grad.norm(dim=-1).flatten() # shape: [num_anchors, 10]
            #                     , k=k, largest=False)[1].squeeze()
            #indices = torch.topk(torch.abs(self.offset_gradient_accum.flatten()), k=k, largest=False)[1].squeeze()
            indices = torch.topk(self._anchor_feat.grad.norm(dim=-1) 
                                 , k=k, largest = False)[1].squeeze()
        
        anchor_mask = torch.ones([self._anchor.shape[0]], device='cuda').bool()
        anchor_mask[indices] = False
            
        # offset_mask = torch.ones([self._anchor.shape[0] * self.n_offsets], device='cuda').bool()
        # offset_mask[indices] = False
        # offset_mask = offset_mask.reshape([self._anchor.shape[0], self.n_offsets])

        # anchor_mask = torch.ones([self._anchor.shape[0]], device='cuda').bool()
        # to_zero = (offset_mask.sum(dim=-1) == 0)
        # anchor_mask[to_zero] = 0
        
        
        # print(f"Removed {k} offsets and {to_zero.sum()} anchors")
        
        # return anchor_mask, offset_mask
        print(f"Removed {(anchor_mask == 0).sum()} anchors")
        return anchor_mask, anchor_mask
    
    
    def save_npz(
        self,
        path,
        compress: bool = True,
        half_precision: bool = True,
        
    ):
        with torch.no_grad():
            self._sort_morton()
            if isinstance(path, str):
                mkdir_p(os.path.dirname(os.path.abspath(path)))

            dtype = torch.half if half_precision else torch.float32

            save_dict = dict()

            save_dict["quantization"] = self.quantization

            # save anchors
            if self.quantization:
                save_dict["anchor"] = self.get_anchor.detach().half().cpu().numpy()
            else:
                save_dict["anchor"] = self._xyz.detach().cpu().numpy()

            # save feat
            if self.quantization:
                
                features = torch.quantize_per_tensor(self._anchor_feat,
                                                    self._anchor_feat_qa.scale,
                                                    self._anchor_feat_qa.zero_point,
                                                    self._anchor_feat_qa.dtype).int_repr()
                save_dict["features"] = features.cpu().numpy()
                save_dict["features_scale"] = self._anchor_feat_qa.scale.cpu().numpy()
                save_dict["features_zero_point"] = self._anchor_feat_qa.zero_point.cpu().numpy()
        
            else:
                save_dict["features"] = self._anchor_feat.detach().cpu().numpy()


            # save offsets
            if self.quantization:
                save_dict["offset"] = self.get_offset.detach().half().cpu().numpy()
                # offset = torch.quantize_per_tensor(self._offset,
                #                                     self._offset_qa.scale,
                #                                     self._offset_qa.zero_point,
                #                                     self._offset_qa.dtype).int_repr()
                # save_dict["offset"] = offset.cpu().numpy()
                # save_dict["offset_scale"] = self._offset_qa.scale.cpu().numpy()
                # save_dict["offset_zero_point"] = self._offset_qa.zero_point.cpu().numpy()  
                
            else:
                save_dict["offset"] = self._offset.detach().to(dtype).cpu().numpy()


            # save scaling
            if self.quantization:
                save_dict["scaling"] = self._scaling.detach().half().cpu().numpy()
                # scaling = torch.quantize_per_tensor(self._scaling,
                #                                     self._scaling_qa.scale,
                #                                     self._scaling_qa.zero_point,
                #                                     self._scaling_qa.dtype).int_repr()
                # save_dict["scaling"] = scaling.cpu().numpy()
                # save_dict["scaling_scale"] = self._scaling_qa.scale.cpu().numpy()
                # save_dict["scaling_zero_point"] = self._scaling_qa.zero_point.cpu().numpy()  
            else:
                save_dict["scaling"] = self._scaling.detach().to(dtype).cpu().numpy()

            # save_fn = np.savez_compressed if compress else np.savez
            # save_fn(path, **save_dict)
        serialized_data = pickle.dumps(save_dict)
        compressed_data = lzma.compress(serialized_data)

        compressed_size_bytes = len(compressed_data)

        compressed_size_mb = compressed_size_bytes / (1024 * 1024)  # Convert bytes to MB

        print(f"Size of the compressed data: {compressed_size_mb:.2f} MB")
        
        with open(path, 'wb') as file:
            file.write(compressed_data)
            
        return compressed_size_mb
            
            
    def load_npz(self, path, override_quantization=False):
        # state_dict = np.load(path)
        
        
        with open(path, 'rb') as file:
            compressed_data = file.read()

        serialized_data = lzma.decompress(compressed_data)
        state_dict = pickle.loads(serialized_data)
        
        quantization = state_dict["quantization"]
        if not override_quantization and self.quantization != quantization:
            print("WARNING: model is not quantisation aware but loaded model is")
        if override_quantization:
            self.quantization = quantization

        # load position
        self._anchor = nn.Parameter(
            torch.from_numpy(state_dict["anchor"]).float().cuda(), requires_grad=True
        )

        # load features
        if quantization:
            
            features_q = torch.from_numpy(state_dict["features"]).int().cuda()
            features_scale = torch.from_numpy(state_dict["features_scale"]).cuda()
            features_zero_point = torch.from_numpy(state_dict["features_zero_point"]).cuda()
            
            features = (features_q - features_zero_point) * features_scale
            self._anchor_feat = nn.Parameter(features, requires_grad=True)
            self._anchor_feat_qa.scale = features_scale
            self._anchor_feat_qa.zero = features_zero_point
            self._anchor_feat_qa.activation_post_process.min_val = features.min()
            self._anchor_feat_qa.activation_post_process.max_val = features.max()
     

        else:
            features = torch.from_numpy(state_dict["features"]).float().cuda()
            self._anchor_feat = nn.Parameter(features, requires_grad=True)

        # load opacity
        if quantization:
            # offset_q = torch.from_numpy(state_dict["offset"]).int().cuda()
            # offset_scale = torch.from_numpy(state_dict["offset_scale"]).cuda()
            # offset_zero_point = torch.from_numpy(
            #     state_dict["offset_zero_point"]
            # ).cuda()
            # offset = (offset_q - offset_zero_point) * offset_scale
            # self._offset = nn.Parameter(
            #     offset, requires_grad=True
            # )
            # self._offset_qa.scale = offset_scale
            # self._offset_qa.zero_point = offset_zero_point
            # self._offset_qa.activation_post_process.min_val = offset.min()
            # self._offset_qa.activation_post_process.max_val = offset.max()
            self._offset = nn.Parameter(
                torch.from_numpy(state_dict["offset"]).float().cuda(), requires_grad=True
            )
        else:
            self._offset = nn.Parameter(
                torch.from_numpy(state_dict["offset"]).float().cuda(),
                requires_grad=True,
            )

        # load scaling
        if quantization:
            # scaling_q = torch.from_numpy(state_dict["scaling"]).int().cuda()
            # scaling_scale = torch.from_numpy(state_dict["scaling_scale"]).cuda()
            # scaling_zero_point = torch.from_numpy(
            #     state_dict["scaling_zero_point"]
            # ).cuda()
            # scaling = (scaling_q - scaling_zero_point) * scaling_scale
            # self._scaling = nn.Parameter(scaling, requires_grad=True)
            # self._scaling_qa.scale = scaling_scale
            # self._scaling_qa.zero_point = scaling_zero_point
            # self._scaling_qa.activation_post_process.min_val = scaling.min()
            # self._scaling_qa.activation_post_process.max_val = scaling.max()
            self._scaling = nn.Parameter(
                torch.from_numpy(state_dict["scaling"]).float().cuda(), requires_grad=True
            )
        else:
            self._scaling = nn.Parameter(
                torch.from_numpy(state_dict["scaling"]).float().cuda(),
                requires_grad=True,
            )
        self._rotation =  torch.zeros((self._anchor.shape[0], 4), device="cuda")
        self._rotation[:, 0] = 1

        
    @torch.no_grad()
    def create_zstd_dictionary_from_quantized_gaussians(
        self,
        dict_size=262_144,
        save_path=None,
        block_size=1024,
        max_blocks=2048,
        small_chunk_gauss=128,
        max_samples=80_000,
        target_corpus_mult=400,
        seed=1234,
    ):

        rng = np.random.default_rng(seed)
        self._sort_morton()  
        anchor_feat_q = torch.quantize_per_tensor(
            self._anchor_feat.detach(),
            self._anchor_feat_qa.scale,
            self._anchor_feat_qa.zero_point,
            self._anchor_feat_qa.dtype,
        ).int_repr().cpu().numpy()

        anchor = np.ascontiguousarray(self._anchor.half().detach().cpu().numpy())
        offset = np.ascontiguousarray(self._offset.half().detach().cpu().numpy())
        scaling = np.ascontiguousarray(self._scaling.half().detach().cpu().numpy())

        N = anchor.shape[0]
        total_blocks = max(1, N // block_size)

        if total_blocks > max_blocks:
            step = max(1, total_blocks // max_blocks)
            selected_blocks = np.arange(0, total_blocks, step, dtype=np.int64)
        else:
            selected_blocks = np.arange(0, total_blocks, dtype=np.int64)

        samples = []
        for i in selected_blocks:
            s = i * block_size
            e = min((i + 1) * block_size, N)
            for off in range(s, e, small_chunk_gauss):
                ss, ee = off, min(off + small_chunk_gauss, e)
                save_dict = {
                    "anchor_feat": anchor_feat_q[ss:ee],
                    "anchor_feat_scale": self._anchor_feat_qa.scale.cpu().numpy(),
                    "anchor_feat_zero_point": self._anchor_feat_qa.zero_point.cpu().numpy(),
                    "anchor": anchor[ss:ee],
                    "offset": offset[ss:ee],
                    "scaling": scaling[ss:ee],
                    "quantization": True,
                }
                samples.append(pickle.dumps(save_dict, protocol=pickle.HIGHEST_PROTOCOL))
                if len(samples) >= max_samples:
                    break
            if len(samples) >= max_samples:
                break

        if len(samples) < 100:
            raise ValueError("Training corpus troppo piccolo per un dizionario utile.")

        byte_cap = dict_size * int(target_corpus_mult)
        capped, total = [], 0
        rng.shuffle(samples)
        for s in samples:
            L = len(s)
            if total + L > byte_cap:
                break
            capped.append(s)
            total += L
        if not capped:
            capped = samples 

        try:
            zstd_dict = zstd.train_dictionary(
                dict_size, capped, k=2048, d=8, steps=8,
                split_point=0.75, shrink_dict=True, threads=0
            )
        except TypeError:
            zstd_dict = zstd.train_dictionary(dict_size, capped)

        if save_path is not None:
            with open(save_path, "wb") as f:
                f.write(zstd_dict.as_bytes())

        return zstd_dict


    def _make_cctx(self, level, zdict=None):

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
            return zstd.ZstdCompressor(level=level, dict_data=zdict, threads=-1, write_content_size=False)


    def save_zstd(self, path, zstd_dict=None, compression_level=22, sort_morton=True):
        """
        Serializza (pickle) un unico payload con feature quantizzate + geom half;
        comprime con Zstd (+dizionario, se fornito).
        """
        if isinstance(path, str):
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        save_dict = dict()
        anchor_feat_q = torch.quantize_per_tensor(
            self._anchor_feat.detach(),
            self._anchor_feat_qa.scale,
            self._anchor_feat_qa.zero_point,
            self._anchor_feat_qa.dtype,
        ).int_repr().cpu().numpy()
        save_dict["anchor_feat"] = anchor_feat_q
        save_dict["anchor_feat_scale"] = self._anchor_feat_qa.scale.cpu().numpy()
        save_dict["anchor_feat_zero_point"] = self._anchor_feat_qa.zero_point.cpu().numpy()
        save_dict["anchor"] = np.ascontiguousarray(self._anchor.half().detach().cpu().numpy())
        save_dict["offset"] = np.ascontiguousarray(self._offset.half().detach().cpu().numpy())
        save_dict["scaling"] = np.ascontiguousarray(self._scaling.half().detach().cpu().numpy())

        serialized_data = pickle.dumps(save_dict, protocol=pickle.HIGHEST_PROTOCOL)


        if isinstance(zstd_dict, (bytes, bytearray)):
            zstd_dict = zstd.ZstdCompressionDict(zstd_dict)
        cctx = self._make_cctx(compression_level, zstd_dict)

        compressed_data = cctx.compress(serialized_data)

        with open(path, "wb") as f:
            f.write(compressed_data)

        size_mb = len(compressed_data) / (1024 * 1024)
        print(f"[ZSTD] Saved {path} ({size_mb:.2f} MB)")
        return size_mb


    def load_zstd(self, path, zstd_dict_path=None):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        zdict = None
        if zstd_dict_path is not None:
            with open(zstd_dict_path, "rb") as f:
                blob = f.read()
            try:
                maybe = pickle.loads(blob)
                if isinstance(maybe, (bytes, bytearray)):
                    zdict = zstd.ZstdCompressionDict(maybe)
                elif isinstance(maybe, dict) and "features" in maybe:
                    zdict = zstd.ZstdCompressionDict(maybe["features"])
                else:
                    zdict = zstd.ZstdCompressionDict(blob)
            except Exception:
                zdict = zstd.ZstdCompressionDict(blob)

        dctx = zstd.ZstdDecompressor(dict_data=zdict) if zdict is not None else zstd.ZstdDecompressor()

        with open(path, "rb") as f:
            comp = f.read()

        import io
        bio = io.BytesIO(comp)
        out_chunks = []
        with dctx.stream_reader(bio) as r:
            while True:
                c = r.read(131072)
                if not c:
                    break
                out_chunks.append(c)
        serialized_data = b"".join(out_chunks)

        data = pickle.loads(serialized_data)

        af_q  = torch.from_numpy(np.ascontiguousarray(data["anchor_feat"])).to(torch.int32).to(device)
        af_s  = torch.from_numpy(np.ascontiguousarray(data["anchor_feat_scale"])).to(torch.float32).to(device)
        af_zp = torch.from_numpy(np.ascontiguousarray(data["anchor_feat_zero_point"])).to(torch.int32).to(device)
        if af_s.ndim == 0:  af_s = af_s.view(1)
        if af_zp.ndim == 0: af_zp = af_zp.view(1)

        anchor_feat = (af_q - af_zp.to(af_q.dtype)) * af_s
        self._anchor_feat = nn.Parameter(anchor_feat.float(), requires_grad=True)
        if hasattr(self, "_anchor_feat_qa"):
            self._anchor_feat_qa.scale = af_s
            self._anchor_feat_qa.zero_point = af_zp
            if hasattr(self._anchor_feat_qa, "activation_post_process"):
                ap = self._anchor_feat_qa.activation_post_process
                ap.min_val = anchor_feat.min()
                ap.max_val = anchor_feat.max()

        self._anchor  = torch.from_numpy(np.ascontiguousarray(data["anchor"])).to(device).half()
        self._offset  = torch.from_numpy(np.ascontiguousarray(data["offset"])).to(device).half()
        self._scaling = torch.from_numpy(np.ascontiguousarray(data["scaling"])).to(device).half()

        return True
        
    def _sort_morton(self):
        def mortonEncode(pos: torch.Tensor) -> torch.Tensor:
            def splitBy3(a):
                x = a & 0x1FFFFF  # we only look at the first 21 bits
                x = (x | x << 32) & 0x1F00000000FFFF
                x = (x | x << 16) & 0x1F0000FF0000FF
                x = (x | x << 8) & 0x100F00F00F00F00F
                x = (x | x << 4) & 0x10C30C30C30C30C3
                x = (x | x << 2) & 0x1249249249249249
                return x
            x, y, z = pos.unbind(-1)
            answer = torch.zeros(len(pos), dtype=torch.long, device=pos.device)
            answer |= splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2
            return answer
        with torch.no_grad():
            xyz_q = (
                (2**21 - 1)
                * (self._anchor - self._anchor.min(0).values)
                / (self._anchor.max(0).values - self._anchor.min(0).values)
            ).long()
            order = mortonEncode(xyz_q).sort().indices
            self._anchor = nn.Parameter(self._anchor[order], requires_grad=True)
            self._anchor_feat = nn.Parameter(self._anchor_feat[order], requires_grad=True)
            self._offset = nn.Parameter(
                self._offset[order], requires_grad=True
            )
            self._scaling = nn.Parameter(self._scaling[order], requires_grad=True)
        return order


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

# import torch
# from functools import reduce
# import numpy as np
# import torch.ao.quantization
# from torch_scatter import scatter_max
# from utils.general_utils import inverse_sigmoid, get_expon_lr_func
# from torch import nn
# import os
# from utils.system_utils import mkdir_p
# from plyfile import PlyData, PlyElement
# from simple_knn._C import distCUDA2
# from utils.graphics_utils import BasicPointCloud
# from utils.general_utils import strip_symmetric, build_scaling_rotation
# from scene.embedding import Embedding



# class FakeQuantizationHalf(torch.autograd.Function):
#     """performs fake quantization for half precision"""

#     @staticmethod
#     def forward(_, x: torch.Tensor) -> torch.Tensor:
#         return x.half().float()

#     @staticmethod
#     def backward(_, grad_output: torch.Tensor) -> torch.Tensor:
#         return grad_output
    
    
# class GaussianModel:

#     def setup_functions(self):
#         def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
#             L = build_scaling_rotation(scaling_modifier * scaling, rotation)
#             actual_covariance = L @ L.transpose(1, 2)
#             symm = strip_symmetric(actual_covariance)
#             return symm
        
#         self.scaling_activation = torch.exp
#         self.scaling_inverse_activation = torch.log

#         self.covariance_activation = build_covariance_from_scaling_rotation

#         self.opacity_activation = torch.sigmoid
#         self.inverse_opacity_activation = inverse_sigmoid

#         self.rotation_activation = torch.nn.functional.normalize


#     def __init__(self, 
#                  feat_dim: int=32, 
#                  n_offsets: int=5, 
#                  voxel_size: float=0.01,
#                  update_depth: int=3, 
#                  update_init_factor: int=100,
#                  update_hierachy_factor: int=4,
#                  use_feat_bank : bool = False,
#                  appearance_dim : int = 32,
#                  ratio : int = 1,
#                  add_opacity_dist : bool = False,
#                  add_cov_dist : bool = False,
#                  add_color_dist : bool = False,
#                  quantization=False
#                  ):

#         self.feat_dim = feat_dim
#         self.n_offsets = n_offsets
#         self.voxel_size = voxel_size
#         self.update_depth = update_depth
#         self.update_init_factor = update_init_factor
#         self.update_hierachy_factor = update_hierachy_factor
#         self.use_feat_bank = use_feat_bank
#         self.quantization = quantization

#         self.appearance_dim = appearance_dim
#         self.embedding_appearance = None
#         self.ratio = ratio
#         self.add_opacity_dist = add_opacity_dist
#         self.add_cov_dist = add_cov_dist
#         self.add_color_dist = add_color_dist

#         self._anchor = torch.empty(0)
#         self._offset = torch.empty(0)
#         self._anchor_feat = torch.empty(0)
        
#         self.opacity_accum = torch.empty(0)

#         self._scaling = torch.empty(0)
#         self._rotation = torch.empty(0)
#         self._opacity = torch.empty(0)
#         self.max_radii2D = torch.empty(0)
        
#         self.offset_gradient_accum = torch.empty(0)
#         self.offset_denom = torch.empty(0)

#         self.anchor_demon = torch.empty(0)
                
#         self.optimizer = None
#         self.percent_dense = 0
#         self.spatial_lr_scale = 0
#         self.setup_functions()

        
#         if self.quantization:
#             #self._anchor_qa = torch.ao.quantization.FakeQuantize(dtype=torch.qint8).cuda()
#             self._anchor_feat_qa = torch.ao.quantization.FakeQuantize(dtype=torch.qint8).cuda()
#             # self._scaling_qa = torch.ao.quantization.FakeQuantize(dytpe=torch.half).cuda()
#             # self._offset_qa = torch.ao.quantization.FakeQuantize(dytpe=torch.half).cuda()
#             self._anchor_qa = FakeQuantizationHalf.apply
#             #self._anchor_feat_qa = FakeQuantizationHalf.apply
#             self._scaling_qa = FakeQuantizationHalf.apply
#             self._offset_qa = FakeQuantizationHalf.apply
        
        
#         if self.use_feat_bank:
#             self.mlp_feature_bank = nn.Sequential(
#                 nn.Linear(3+1, feat_dim),
#                 nn.ReLU(True),
#                 nn.Linear(feat_dim, 3),
#                 nn.Softmax(dim=1)
#             ).cuda()

#         self.opacity_dist_dim = 1 if self.add_opacity_dist else 0
#         self.mlp_opacity = nn.Sequential(
#             nn.Linear(feat_dim+3+self.opacity_dist_dim, feat_dim),
#             nn.ReLU(True),
#             nn.Linear(feat_dim, n_offsets),
#             nn.Tanh()
#         ).cuda()

#         self.add_cov_dist = add_cov_dist
#         self.cov_dist_dim = 1 if self.add_cov_dist else 0
#         self.mlp_cov = nn.Sequential(
#             nn.Linear(feat_dim+3+self.cov_dist_dim, feat_dim),
#             nn.ReLU(True),
#             nn.Linear(feat_dim, 7*self.n_offsets),
#         ).cuda()

#         self.color_dist_dim = 1 if self.add_color_dist else 0
#         self.mlp_color = nn.Sequential(
#             nn.Linear(feat_dim+3+self.color_dist_dim+self.appearance_dim, feat_dim),
#             nn.ReLU(True),
#             nn.Linear(feat_dim, 3*self.n_offsets),
#             nn.Sigmoid()
#         ).cuda()


#     def eval(self):
#         self.mlp_opacity.eval()
#         self.mlp_cov.eval()
#         self.mlp_color.eval()
#         if self.appearance_dim > 0:
#             self.embedding_appearance.eval()
#         if self.use_feat_bank:
#             self.mlp_feature_bank.eval()

#     def train(self):
#         self.mlp_opacity.train()
#         self.mlp_cov.train()
#         self.mlp_color.train()
#         if self.appearance_dim > 0:
#             self.embedding_appearance.train()
#         if self.use_feat_bank:                   
#             self.mlp_feature_bank.train()

#     def capture(self):
#         return (
#             self._anchor,
#             self._anchor_feat,
#             self._offset,
#             #self._local,
#             self._scaling,
#             self._rotation,
#             self._opacity,
#             self.max_radii2D,
#             #self.denom,
#             self.optimizer.state_dict(),
#             self.spatial_lr_scale,
#         )
    
#     def restore(self, model_args, training_args):
#         (
#         self._anchor, 
#         self._anchor_feat,
#         self._offset,
#         self._scaling, 
#         self._rotation, 
#         self._opacity,
#         self.max_radii2D,
#         opt_dict, 
#         self.spatial_lr_scale) = model_args
#         self.training_setup(training_args)

#         self.optimizer.load_state_dict(opt_dict)


#     def set_appearance(self, num_cameras):
#         if self.appearance_dim > 0:
#             self.embedding_appearance = Embedding(num_cameras, self.appearance_dim).cuda()
            
#     @property
#     def get_anchor_feat(self):
#         if self.quantization:
#             return self._anchor_feat_qa(self._anchor_feat)
#         return self._anchor_feat
    
#     @property
#     def get_offset(self):
#         if self.quantization:
#             return self._offset_qa(self._offset)
#         return self._offset

#     @property
#     def get_appearance(self):
#         return self.embedding_appearance

#     @property
#     def get_scaling(self):
#         if self.quantization:
#             return self._scaling_qa(1.0*self.scaling_activation(self._scaling))
        
#         return 1.0*self.scaling_activation(self._scaling)
    
#     @property
#     def get_featurebank_mlp(self):
#         return self.mlp_feature_bank
    
#     @property
#     def get_opacity_mlp(self):
#         return self.mlp_opacity
    
#     @property
#     def get_cov_mlp(self):
#         return self.mlp_cov

#     @property
#     def get_color_mlp(self):
#         return self.mlp_color
    
#     @property
#     def get_rotation(self):
#         return self.rotation_activation(self._rotation)
    
#     @property
#     def get_anchor(self):
#         if self.quantization:
#             return self._anchor_qa(self._anchor)
#         return self._anchor
    
#     @property
#     def set_anchor(self, new_anchor):
#         assert self._anchor.shape == new_anchor.shape
#         del self._anchor
#         torch.cuda.empty_cache()
#         self._anchor = new_anchor
    
#     @property
#     def get_opacity(self):
#         return self.opacity_activation(self._opacity)
    
#     def get_covariance(self, scaling_modifier = 1):
#         return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
#     def voxelize_sample(self, data=None, voxel_size=0.01):
#         np.random.shuffle(data)
#         data = np.unique(np.round(data/voxel_size), axis=0)*voxel_size
        
#         return data

#     def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
#         self.spatial_lr_scale = spatial_lr_scale
#         points = pcd.points[::self.ratio]

#         if self.voxel_size <= 0:
#             init_points = torch.tensor(points).float().cuda()
#             init_dist = distCUDA2(init_points).float().cuda()
#             median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))
#             self.voxel_size = median_dist.item()
#             del init_dist
#             del init_points
#             torch.cuda.empty_cache()

#         print(f'Initial voxel_size: {self.voxel_size}')
        
        
#         points = self.voxelize_sample(points, voxel_size=self.voxel_size)
#         fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
#         offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
#         anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()
        
#         print("Number of points at initialisation : ", fused_point_cloud.shape[0])

#         dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 0.0000001)
#         scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 6)
        
#         rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
#         rots[:, 0] = 1

#         opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

#         self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
#         self._offset = nn.Parameter(offsets.requires_grad_(True))
#         self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
#         self._scaling = nn.Parameter(scales.requires_grad_(True))
#         self._rotation = nn.Parameter(rots.requires_grad_(False))
#         self._opacity = nn.Parameter(opacities.requires_grad_(False))
#         self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")


#     def training_setup(self, training_args):
#         self.percent_dense = training_args.percent_dense

#         self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

#         self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
#         self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
#         self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        
        
#         if self.use_feat_bank:
#             l = [
#                 {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
#                 {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
#                 {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
#                 {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
#                 {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
#                 {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                
#                 {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
#                 {'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"},
#                 {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
#                 {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
#                 {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"},
#             ]
#         elif self.appearance_dim > 0:
#             l = [
#                 {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
#                 {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
#                 {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
#                 {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
#                 {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
#                 {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

#                 {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
#                 {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
#                 {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
#                 {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"},
#             ]
#         else:
#             l = [
#                 {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
#                 {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
#                 {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
#                 {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
#                 {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
#                 {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

#                 {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
#                 {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
#                 {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
#             ]

#         self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
#         self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
#                                                     lr_final=training_args.position_lr_final*self.spatial_lr_scale,
#                                                     lr_delay_mult=training_args.position_lr_delay_mult,
#                                                     max_steps=training_args.position_lr_max_steps)
#         self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
#                                                     lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
#                                                     lr_delay_mult=training_args.offset_lr_delay_mult,
#                                                     max_steps=training_args.offset_lr_max_steps)
        
#         self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
#                                                     lr_final=training_args.mlp_opacity_lr_final,
#                                                     lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
#                                                     max_steps=training_args.mlp_opacity_lr_max_steps)
        
#         self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
#                                                     lr_final=training_args.mlp_cov_lr_final,
#                                                     lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
#                                                     max_steps=training_args.mlp_cov_lr_max_steps)
        
#         self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
#                                                     lr_final=training_args.mlp_color_lr_final,
#                                                     lr_delay_mult=training_args.mlp_color_lr_delay_mult,
#                                                     max_steps=training_args.mlp_color_lr_max_steps)
#         if self.use_feat_bank:
#             self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
#                                                         lr_final=training_args.mlp_featurebank_lr_final,
#                                                         lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
#                                                         max_steps=training_args.mlp_featurebank_lr_max_steps)
#         if self.appearance_dim > 0:
#             self.appearance_scheduler_args = get_expon_lr_func(lr_init=training_args.appearance_lr_init,
#                                                         lr_final=training_args.appearance_lr_final,
#                                                         lr_delay_mult=training_args.appearance_lr_delay_mult,
#                                                         max_steps=training_args.appearance_lr_max_steps)

#     def update_learning_rate(self, iteration):
#         ''' Learning rate scheduling per step '''
#         for param_group in self.optimizer.param_groups:
#             if param_group["name"] == "offset":
#                 lr = self.offset_scheduler_args(iteration)
#                 param_group['lr'] = lr
#             if param_group["name"] == "anchor":
#                 lr = self.anchor_scheduler_args(iteration)
#                 param_group['lr'] = lr
#             if param_group["name"] == "mlp_opacity":
#                 lr = self.mlp_opacity_scheduler_args(iteration)
#                 param_group['lr'] = lr
#             if param_group["name"] == "mlp_cov":
#                 lr = self.mlp_cov_scheduler_args(iteration)
#                 param_group['lr'] = lr
#             if param_group["name"] == "mlp_color":
#                 lr = self.mlp_color_scheduler_args(iteration)
#                 param_group['lr'] = lr
#             if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
#                 lr = self.mlp_featurebank_scheduler_args(iteration)
#                 param_group['lr'] = lr
#             if self.appearance_dim > 0 and param_group["name"] == "embedding_appearance":
#                 lr = self.appearance_scheduler_args(iteration)
#                 param_group['lr'] = lr
            
            
#     def construct_list_of_attributes(self):
#         l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
#         for i in range(self._offset.shape[1]*self._offset.shape[2]):
#             l.append('f_offset_{}'.format(i))
#         for i in range(self._anchor_feat.shape[1]):
#             l.append('f_anchor_feat_{}'.format(i))
#         l.append('opacity')
#         for i in range(self._scaling.shape[1]):
#             l.append('scale_{}'.format(i))
#         for i in range(self._rotation.shape[1]):
#             l.append('rot_{}'.format(i))
#         return l

#     def save_ply(self, path):
#         mkdir_p(os.path.dirname(path))

#         anchor = self._anchor.detach().cpu().numpy()
#         normals = np.zeros_like(anchor)
#         anchor_feat = self._anchor_feat.detach().cpu().numpy()
#         offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
#         opacities = self._opacity.detach().cpu().numpy()
#         scale = self._scaling.detach().cpu().numpy()
#         rotation = self._rotation.detach().cpu().numpy()

#         dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

#         elements = np.empty(anchor.shape[0], dtype=dtype_full)
#         attributes = np.concatenate((anchor, normals, offset, anchor_feat, opacities, scale, rotation), axis=1)
#         elements[:] = list(map(tuple, attributes))
#         el = PlyElement.describe(elements, 'vertex')
#         PlyData([el]).write(path)

#     def load_ply_sparse_gaussian(self, path):
#         plydata = PlyData.read(path)

#         anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
#                         np.asarray(plydata.elements[0]["y"]),
#                         np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
#         opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

#         scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
#         scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
#         scales = np.zeros((anchor.shape[0], len(scale_names)))
#         for idx, attr_name in enumerate(scale_names):
#             scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

#         rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
#         rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
#         rots = np.zeros((anchor.shape[0], len(rot_names)))
#         for idx, attr_name in enumerate(rot_names):
#             rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        
#         # anchor_feat
#         anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
#         anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
#         anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
#         for idx, attr_name in enumerate(anchor_feat_names):
#             anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

#         offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
#         offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
#         offsets = np.zeros((anchor.shape[0], len(offset_names)))
#         for idx, attr_name in enumerate(offset_names):
#             offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
#         offsets = offsets.reshape((offsets.shape[0], 3, -1))
        
#         self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))

#         self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
#         self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
#         self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
#         self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
#         self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))


#     def replace_tensor_to_optimizer(self, tensor, name):
#         optimizable_tensors = {}
#         for group in self.optimizer.param_groups:
#             if group["name"] == name:
#                 stored_state = self.optimizer.state.get(group['params'][0], None)
#                 stored_state["exp_avg"] = torch.zeros_like(tensor)
#                 stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

#                 del self.optimizer.state[group['params'][0]]
#                 group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
#                 self.optimizer.state[group['params'][0]] = stored_state

#                 optimizable_tensors[group["name"]] = group["params"][0]
#         return optimizable_tensors


#     def cat_tensors_to_optimizer(self, tensors_dict):
#         optimizable_tensors = {}
#         for group in self.optimizer.param_groups:
#             if  'mlp' in group['name'] or \
#                 'conv' in group['name'] or \
#                 'feat_base' in group['name'] or \
#                 'embedding' in group['name']:
#                 continue
#             assert len(group["params"]) == 1
#             extension_tensor = tensors_dict[group["name"]]
#             stored_state = self.optimizer.state.get(group['params'][0], None)
#             if stored_state is not None:
#                 stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
#                 stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

#                 del self.optimizer.state[group['params'][0]]
#                 group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
#                 self.optimizer.state[group['params'][0]] = stored_state

#                 optimizable_tensors[group["name"]] = group["params"][0]
#             else:
#                 group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
#                 optimizable_tensors[group["name"]] = group["params"][0]

#         return optimizable_tensors


#     # statis grad information to guide liftting. 
#     def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask):
#         # update opacity stats
#         temp_opacity = opacity.clone().view(-1).detach()
#         temp_opacity[temp_opacity<0] = 0
        
#         temp_opacity = temp_opacity.view([-1, self.n_offsets])
#         self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)
        
#         # update anchor visiting statis
#         self.anchor_demon[anchor_visible_mask] += 1

#         # update neural gaussian statis
#         anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
#         combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
#         combined_mask[anchor_visible_mask] = offset_selection_mask
#         temp_mask = combined_mask.clone()
#         combined_mask[temp_mask] = update_filter
        
#         grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
#         self.offset_gradient_accum[combined_mask] += grad_norm
#         self.offset_denom[combined_mask] += 1

        

        
#     def _prune_anchor_optimizer(self, mask):
#         optimizable_tensors = {}
#         for group in self.optimizer.param_groups:
#             if  'mlp' in group['name'] or \
#                 'conv' in group['name'] or \
#                 'feat_base' in group['name'] or \
#                 'embedding' in group['name']:
#                 continue

#             stored_state = self.optimizer.state.get(group['params'][0], None)
#             if stored_state is not None:
#                 stored_state["exp_avg"] = stored_state["exp_avg"][mask]
#                 stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

#                 del self.optimizer.state[group['params'][0]]
#                 group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
#                 self.optimizer.state[group['params'][0]] = stored_state
#                 if group['name'] == "scaling":
#                     scales = group["params"][0]
#                     temp = scales[:,3:]
#                     temp[temp>0.05] = 0.05
#                     group["params"][0][:,3:] = temp
#                 optimizable_tensors[group["name"]] = group["params"][0]
#             else:
#                 group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
#                 if group['name'] == "scaling":
#                     scales = group["params"][0]
#                     temp = scales[:,3:]
#                     temp[temp>0.05] = 0.05
#                     group["params"][0][:,3:] = temp
#                 optimizable_tensors[group["name"]] = group["params"][0]
            
            
#         return optimizable_tensors

#     def prune_anchor(self,mask):
#         valid_points_mask = ~mask

#         optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

#         self._anchor = optimizable_tensors["anchor"]
#         self._offset = optimizable_tensors["offset"]
#         self._anchor_feat = optimizable_tensors["anchor_feat"]
#         self._opacity = optimizable_tensors["opacity"]
#         self._scaling = optimizable_tensors["scaling"]
#         self._rotation = optimizable_tensors["rotation"]

    
#     def anchor_growing(self, grads, threshold, offset_mask):
#         ## 
#         init_length = self.get_anchor.shape[0]*self.n_offsets
#         for i in range(self.update_depth):
#             # update threshold
#             cur_threshold = threshold*((self.update_hierachy_factor//2)**i)
#             # mask from grad threshold
#             candidate_mask = (grads >= cur_threshold)
#             candidate_mask = torch.logical_and(candidate_mask, offset_mask)
            
#             # random pick
#             rand_mask = torch.rand_like(candidate_mask.float())>(0.5**(i+1))
#             rand_mask = rand_mask.cuda()
#             candidate_mask = torch.logical_and(candidate_mask, rand_mask)
            
#             length_inc = self.get_anchor.shape[0]*self.n_offsets - init_length
#             if length_inc == 0:
#                 if i > 0:
#                     continue
#             else:
#                 candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)

#             all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:,:3].unsqueeze(dim=1)
            
#             # assert self.update_init_factor // (self.update_hierachy_factor**i) > 0
#             # size_factor = min(self.update_init_factor // (self.update_hierachy_factor**i), 1)
#             size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
#             cur_size = self.voxel_size*size_factor
            
#             grid_coords = torch.round(self.get_anchor / cur_size).int()

#             selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
#             selected_grid_coords = torch.round(selected_xyz / cur_size).int()

#             selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)


#             ## split data for reducing peak memory calling
#             use_chunk = True
#             if use_chunk:
#                 chunk_size = 4096
#                 max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
#                 remove_duplicates_list = []
#                 for i in range(max_iters):
#                     cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
#                     remove_duplicates_list.append(cur_remove_duplicates)
                
#                 remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
#             else:
#                 remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)

#             remove_duplicates = ~remove_duplicates
#             candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size

            
#             if candidate_anchor.shape[0] > 0:
#                 new_scaling = torch.ones_like(candidate_anchor).repeat([1,2]).float().cuda()*cur_size # *0.05
#                 new_scaling = torch.log(new_scaling)
#                 new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
#                 new_rotation[:,0] = 1.0

#                 new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))

#                 new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]

#                 new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]

#                 new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()

#                 d = {
#                     "anchor": candidate_anchor,
#                     "scaling": new_scaling,
#                     "rotation": new_rotation,
#                     "anchor_feat": new_feat,
#                     "offset": new_offsets,
#                     "opacity": new_opacities,
#                 }
                

#                 temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
#                 del self.anchor_demon
#                 self.anchor_demon = temp_anchor_demon

#                 temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
#                 del self.opacity_accum
#                 self.opacity_accum = temp_opacity_accum

#                 torch.cuda.empty_cache()
                
#                 optimizable_tensors = self.cat_tensors_to_optimizer(d)
#                 self._anchor = optimizable_tensors["anchor"]
#                 self._scaling = optimizable_tensors["scaling"]
#                 self._rotation = optimizable_tensors["rotation"]
#                 self._anchor_feat = optimizable_tensors["anchor_feat"]
#                 self._offset = optimizable_tensors["offset"]
#                 self._opacity = optimizable_tensors["opacity"]
                


#     def adjust_anchor(self, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005, grow=True, prune=True):
#         # # adding anchors
#         grads = self.offset_gradient_accum / self.offset_denom # [N*k, 1]
#         grads[grads.isnan()] = 0.0
#         grads_norm = torch.norm(grads, dim=-1)
#         offset_mask = (self.offset_denom > check_interval*success_threshold*0.5).squeeze(dim=1)
        
#         if grow:
#             self.anchor_growing(grads_norm, grad_threshold, offset_mask)
        
#         # update offset_denom
#         self.offset_denom[offset_mask] = 0
#         padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
#                                            dtype=torch.int32, 
#                                            device=self.offset_denom.device)
#         self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

#         self.offset_gradient_accum[offset_mask] = 0
#         padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
#                                            dtype=torch.int32, 
#                                            device=self.offset_gradient_accum.device)
#         self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)
#         if prune:
#             # # prune anchors
#             prune_mask = (self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1)
#             anchors_mask = (self.anchor_demon > check_interval*success_threshold).squeeze(dim=1) # [N, 1]
#             prune_mask = torch.logical_and(prune_mask, anchors_mask) # [N] 
            
#             # update offset_denom
#             offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
#             offset_denom = offset_denom.view([-1, 1])
#             del self.offset_denom
#             self.offset_denom = offset_denom

#             offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
#             offset_gradient_accum = offset_gradient_accum.view([-1, 1])
#             del self.offset_gradient_accum
#             self.offset_gradient_accum = offset_gradient_accum
            
#             # update opacity accum 
#             if anchors_mask.sum()>0:
#                 self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
#                 self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            
#             temp_opacity_accum = self.opacity_accum[~prune_mask]
#             del self.opacity_accum
#             self.opacity_accum = temp_opacity_accum

#             temp_anchor_demon = self.anchor_demon[~prune_mask]
#             del self.anchor_demon
#             self.anchor_demon = temp_anchor_demon

#             if prune_mask.shape[0]>0:
#                 self.prune_anchor(prune_mask)
        
#         self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")
#         print('Number of anchors:', self._anchor.shape[0])

#     def save_mlp_checkpoints(self, path, mode = 'unite'):#split or unite
#         mkdir_p(os.path.dirname(path))
#         if mode == 'split':
#             self.mlp_opacity.eval()
#             opacity_mlp = torch.jit.trace(self.mlp_opacity, (torch.rand(1, self.feat_dim+3+self.opacity_dist_dim).cuda()))
#             opacity_mlp.save(os.path.join(path, 'opacity_mlp.pt'))
#             self.mlp_opacity.train()

#             self.mlp_cov.eval()
#             cov_mlp = torch.jit.trace(self.mlp_cov, (torch.rand(1, self.feat_dim+3+self.cov_dist_dim).cuda()))
#             cov_mlp.save(os.path.join(path, 'cov_mlp.pt'))
#             self.mlp_cov.train()

#             self.mlp_color.eval()
#             color_mlp = torch.jit.trace(self.mlp_color, (torch.rand(1, self.feat_dim+3+self.color_dist_dim+self.appearance_dim).cuda()))
#             color_mlp.save(os.path.join(path, 'color_mlp.pt'))
#             self.mlp_color.train()

#             if self.use_feat_bank:
#                 self.mlp_feature_bank.eval()
#                 feature_bank_mlp = torch.jit.trace(self.mlp_feature_bank, (torch.rand(1, 3+1).cuda()))
#                 feature_bank_mlp.save(os.path.join(path, 'feature_bank_mlp.pt'))
#                 self.mlp_feature_bank.train()

#             if self.appearance_dim:
#                 self.embedding_appearance.eval()
#                 emd = torch.jit.trace(self.embedding_appearance, (torch.zeros((1,), dtype=torch.long).cuda()))
#                 emd.save(os.path.join(path, 'embedding_appearance.pt'))
#                 self.embedding_appearance.train()

#         elif mode == 'unite':
#             if self.use_feat_bank:
#                 torch.save({
#                     'opacity_mlp': self.mlp_opacity.state_dict(),
#                     'cov_mlp': self.mlp_cov.state_dict(),
#                     'color_mlp': self.mlp_color.state_dict(),
#                     'feature_bank_mlp': self.mlp_feature_bank.state_dict(),
#                     'appearance': self.embedding_appearance.state_dict()
#                     }, os.path.join(path, 'checkpoints.pth'))
#             elif self.appearance_dim > 0:
#                 torch.save({
#                     'opacity_mlp': self.mlp_opacity.state_dict(),
#                     'cov_mlp': self.mlp_cov.state_dict(),
#                     'color_mlp': self.mlp_color.state_dict(),
#                     'appearance': self.embedding_appearance.state_dict()
#                     }, os.path.join(path, 'checkpoints.pth'))
#             else:
#                 torch.save({
#                     'opacity_mlp': self.mlp_opacity.state_dict(),
#                     'cov_mlp': self.mlp_cov.state_dict(),
#                     'color_mlp': self.mlp_color.state_dict(),
#                     }, os.path.join(path, 'checkpoints.pth'))
#         else:
#             raise NotImplementedError


#     def load_mlp_checkpoints(self, path, mode = 'unite'):#split or unite
#         if mode == 'split':
#             self.mlp_opacity = torch.jit.load(os.path.join(path, 'opacity_mlp.pt')).cuda()
#             self.mlp_cov = torch.jit.load(os.path.join(path, 'cov_mlp.pt')).cuda()
#             self.mlp_color = torch.jit.load(os.path.join(path, 'color_mlp.pt')).cuda()
#             if self.use_feat_bank:
#                 self.mlp_feature_bank = torch.jit.load(os.path.join(path, 'feature_bank_mlp.pt')).cuda()
#             if self.appearance_dim > 0:
#                 self.embedding_appearance = torch.jit.load(os.path.join(path, 'embedding_appearance.pt')).cuda()
#         elif mode == 'unite':
#             checkpoint = torch.load(os.path.join(path, 'checkpoints.pth'))
#             self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
#             self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
#             self.mlp_color.load_state_dict(checkpoint['color_mlp'])
#             if self.use_feat_bank:
#                 self.mlp_feature_bank.load_state_dict(checkpoint['feature_bank_mlp'])
#             if self.appearance_dim > 0:
#                 self.embedding_appearance.load_state_dict(checkpoint['appearance'])
#         else:
#             raise NotImplementedError


#     def gradient_prune(self, k, random=False):
#         if random:
#             indices = torch.multinomial(torch.ones((self._xyz.shape[0])), num_samples=k, replacement=False)
#         else:
#             # indices = torch.topk(
#             #                     self._offset.grad.norm(dim=-1).flatten() # shape: [num_anchors, 10]
#             #                     , k=k, largest=False)[1].squeeze()
#             #indices = torch.topk(torch.abs(self.offset_gradient_accum.flatten()), k=k, largest=False)[1].squeeze()
#             indices = torch.topk(self._anchor_feat.grad.norm(dim=-1) 
#                                  , k=k, largest = False)[1].squeeze()
        
#         anchor_mask = torch.ones([self._anchor.shape[0]], device='cuda').bool()
#         anchor_mask[indices] = False
            
#         # offset_mask = torch.ones([self._anchor.shape[0] * self.n_offsets], device='cuda').bool()
#         # offset_mask[indices] = False
#         # offset_mask = offset_mask.reshape([self._anchor.shape[0], self.n_offsets])

#         # anchor_mask = torch.ones([self._anchor.shape[0]], device='cuda').bool()
#         # to_zero = (offset_mask.sum(dim=-1) == 0)
#         # anchor_mask[to_zero] = 0
        
        
#         # print(f"Removed {k} offsets and {to_zero.sum()} anchors")
        
#         # return anchor_mask, offset_mask
#         print(f"Removed {(anchor_mask == 0).sum()} anchors")
#         return anchor_mask, anchor_mask
    
    
#     def save_npz(
#         self,
#         path,
#         mask,
#         compress: bool = True,
#         half_precision: bool = True,
        
#     ):
#         with torch.no_grad():
            
#             if isinstance(path, str):
#                 mkdir_p(os.path.dirname(os.path.abspath(path)))

#             dtype = torch.half if half_precision else torch.float32

#             save_dict = dict()

#             save_dict["quantization"] = self.quantization

#             # save anchors
#             if self.quantization:
#                 save_dict["anchor"] = self.get_anchor.detach()[mask].half().cpu().numpy()
#             else:
#                 save_dict["anchor"] = self._xyz.detach()[mask].cpu().numpy()

#             # save feat
#             if self.quantization:
                
#                 features = torch.quantize_per_tensor(self._anchor_feat,
#                                                     self._anchor_feat_qa.scale,
#                                                     self._anchor_feat_qa.zero_point,
#                                                     self._anchor_feat_qa.dtype).int_repr()[mask]
#                 save_dict["features"] = features.cpu().numpy()
#                 save_dict["features_scale"] = self._anchor_feat_qa.scale.cpu().numpy()
#                 save_dict["features_zero_point"] = self._anchor_feat_qa.zero_point.cpu().numpy()
        
#             else:
#                 save_dict["features"] = self._anchor_feat.detach()[mask].cpu().numpy()


#             # save offsets
#             if self.quantization:
#                 save_dict["offset"] = self.get_offset.detach()[mask].half().cpu().numpy()
#                 # offset = torch.quantize_per_tensor(self._offset,
#                 #                                     self._offset_qa.scale,
#                 #                                     self._offset_qa.zero_point,
#                 #                                     self._offset_qa.dtype).int_repr()
#                 # save_dict["offset"] = offset.cpu().numpy()
#                 # save_dict["offset_scale"] = self._offset_qa.scale.cpu().numpy()
#                 # save_dict["offset_zero_point"] = self._offset_qa.zero_point.cpu().numpy()  
                
#             else:
#                 save_dict["offset"] = self._offset.detach()[mask].to(dtype).cpu().numpy()


#             # save scaling
#             if self.quantization:
#                 save_dict["scaling"] = self._scaling.detach()[mask].half().cpu().numpy()
#                 # scaling = torch.quantize_per_tensor(self._scaling,
#                 #                                     self._scaling_qa.scale,
#                 #                                     self._scaling_qa.zero_point,
#                 #                                     self._scaling_qa.dtype).int_repr()
#                 # save_dict["scaling"] = scaling.cpu().numpy()
#                 # save_dict["scaling_scale"] = self._scaling_qa.scale.cpu().numpy()
#                 # save_dict["scaling_zero_point"] = self._scaling_qa.zero_point.cpu().numpy()  
#             else:
#                 save_dict["scaling"] = self._scaling.detach()[mask].to(dtype).cpu().numpy()

#             save_fn = np.savez_compressed if compress else np.savez
#             save_fn(path, **save_dict)
            
            
#     def load_npz(self, path, override_quantization=False):
#         state_dict = np.load(path)

#         quantization = state_dict["quantization"]
#         if not override_quantization and self.quantization != quantization:
#             print("WARNING: model is not quantisation aware but loaded model is")
#         if override_quantization:
#             self.quantization = quantization

#         # load position
#         self._anchor = nn.Parameter(
#             torch.from_numpy(state_dict["anchor"]).float().cuda(), requires_grad=True
#         )

#         # load features
#         if quantization:
            
#             features_q = torch.from_numpy(state_dict["features"]).int().cuda()
#             features_scale = torch.from_numpy(state_dict["features_scale"]).cuda()
#             features_zero_point = torch.from_numpy(state_dict["features_zero_point"]).cuda()
            
#             features = (features_q - features_zero_point) * features_scale
#             self._anchor_feat = nn.Parameter(features, requires_grad=True)
#             self._anchor_feat_qa.scale = features_scale
#             self._anchor_feat_qa.zero = features_zero_point
#             self._anchor_feat_qa.activation_post_process.min_val = features.min()
#             self._anchor_feat_qa.activation_post_process.max_val = features.max()
     

#         else:
#             features = torch.from_numpy(state_dict["features"]).float().cuda()
#             self._anchor_feat = nn.Parameter(features, requires_grad=True)

#         # load opacity
#         if quantization:
#             # offset_q = torch.from_numpy(state_dict["offset"]).int().cuda()
#             # offset_scale = torch.from_numpy(state_dict["offset_scale"]).cuda()
#             # offset_zero_point = torch.from_numpy(
#             #     state_dict["offset_zero_point"]
#             # ).cuda()
#             # offset = (offset_q - offset_zero_point) * offset_scale
#             # self._offset = nn.Parameter(
#             #     offset, requires_grad=True
#             # )
#             # self._offset_qa.scale = offset_scale
#             # self._offset_qa.zero_point = offset_zero_point
#             # self._offset_qa.activation_post_process.min_val = offset.min()
#             # self._offset_qa.activation_post_process.max_val = offset.max()
#             self._offset = nn.Parameter(
#                 torch.from_numpy(state_dict["offset"]).float().cuda(), requires_grad=True
#             )
#         else:
#             self._offset = nn.Parameter(
#                 torch.from_numpy(state_dict["offset"]).float().cuda(),
#                 requires_grad=True,
#             )

#         # load scaling
#         if quantization:
#             # scaling_q = torch.from_numpy(state_dict["scaling"]).int().cuda()
#             # scaling_scale = torch.from_numpy(state_dict["scaling_scale"]).cuda()
#             # scaling_zero_point = torch.from_numpy(
#             #     state_dict["scaling_zero_point"]
#             # ).cuda()
#             # scaling = (scaling_q - scaling_zero_point) * scaling_scale
#             # self._scaling = nn.Parameter(scaling, requires_grad=True)
#             # self._scaling_qa.scale = scaling_scale
#             # self._scaling_qa.zero_point = scaling_zero_point
#             # self._scaling_qa.activation_post_process.min_val = scaling.min()
#             # self._scaling_qa.activation_post_process.max_val = scaling.max()
#             self._scaling = nn.Parameter(
#                 torch.from_numpy(state_dict["scaling"]).float().cuda(), requires_grad=True
#             )
#         else:
#             self._scaling = nn.Parameter(
#                 torch.from_numpy(state_dict["scaling"]).float().cuda(),
#                 requires_grad=True,
#             )
#         self._rotation =  torch.zeros((self._anchor.shape[0], 4), device="cuda")
#         self._rotation[:, 0] = 1