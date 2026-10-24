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
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from utils.loss_utils import photometric_loss
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
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


    def compute_gradient_importance(self, score_mode="normalized"):
        """Compute per-Gaussian importance from all parameter-gradient groups."""
        parameters = (
            self._opacity,
            self._xyz,
            self._scaling,
            self._rotation,
            self._features_rest,
            self._features_dc,
        )
        names = ('opacity', 'xyz', 'scaling', 'rotation', 'features_rest', 'features_dc')
        missing = [
            name for name, parameter in zip(names, parameters)
            if parameter.grad is None
        ]
        if missing:
            raise RuntimeError(
                f"Missing gradients while computing pruning importance: {', '.join(missing)}"
            )

        contributions = [
            self._opacity.grad.norm(dim=1),
            self._xyz.grad.norm(dim=1),
            self._scaling.grad.norm(dim=1),
            self._rotation.grad.norm(dim=1),
            self._features_rest.grad.norm(dim=-1).norm(dim=-1),
            self._features_dc.grad.norm(dim=-1).norm(dim=-1),
        ]
        if score_mode == 'normalized':
            normalized_contributions = []
            for contribution in contributions:
                epsilon = torch.finfo(contribution.dtype).eps
                rms = contribution.square().mean().sqrt().clamp_min(epsilon)
                normalized_contributions.append(contribution / rms)
            score = torch.stack(normalized_contributions, dim=0).sum(dim=0)
        elif score_mode == 'raw':
            # Preserve the previous left-to-right summation order for reproducibility.
            score = contributions[0]
            for contribution in contributions[1:]:
                score = score + contribution
        else:
            raise ValueError(f'Unknown gradient score mode: {score_mode}')

        score = score.detach()
        if score.ndim != 1 or score.shape[0] != self._xyz.shape[0]:
            raise RuntimeError(
                f'Unexpected gradient score shape {tuple(score.shape)} for '
                f'{self._xyz.shape[0]} Gaussians'
            )
        if not torch.isfinite(score).all():
            raise RuntimeError('Gradient score contains NaN or infinite values')
        return score


    def gradient_prune(self, k, prune_type='gradient_all', largest=False,
                       score_mode='normalized', scores=None):
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
            elif prune_type in ('gradient_all', 'gradient_oneshot'):
                score = scores if scores is not None else self.compute_gradient_importance(score_mode=score_mode)
                indices = torch.topk(score, k=k, largest=largest)[1].squeeze()
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


    # ------------------------------------------------------------------
    # Primitive bookkeeping
    # The scalable core only touches primitives through these methods, which
    # is what keeps it backbone-agnostic.
    # ------------------------------------------------------------------

    def primitive_names(self, include_empty=False):
        """Attributes holding one row per primitive, in a stable order."""
        names = ["_xyz", "_opacity", "_scaling", "_rotation",
                 "_features_dc", "_features_rest"]
        if include_empty:
            return names
        return [n for n in names if getattr(self, n, torch.empty(0)).numel() > 0]

    def index_primitives(self, index, requires_grad=True):
        """Select or reorder primitives along dim 0 (boolean mask or indices)."""
        for name in self.primitive_names():
            tensor = getattr(self, name)
            tensor = tensor.data if isinstance(tensor, nn.Parameter) else tensor
            value = tensor[index.to(tensor.device)].clone()
            setattr(self, name, nn.Parameter(value, requires_grad=requires_grad))

    def append_primitives(self, other):
        """Append another model's primitives (incremental LoD decoding)."""
        for name in self.primitive_names():
            mine, theirs = getattr(self, name), getattr(other, name)
            mine = mine.data if isinstance(mine, nn.Parameter) else mine
            theirs = theirs.data if isinstance(theirs, nn.Parameter) else theirs
            setattr(self, name, torch.cat([mine, theirs.to(mine.device)], dim=0))

    def subset(self, mask, copy_qa=True):
        """A detached copy holding only the selected primitives."""
        clone = self.clone_empty(copy_qa=copy_qa)
        for name in self.primitive_names():
            tensor = getattr(self, name)
            tensor = tensor.data if isinstance(tensor, nn.Parameter) else tensor
            setattr(clone, name, nn.Parameter(tensor[mask].clone(), requires_grad=False))
        return clone

    def clone_empty(self, copy_qa=False):
        """An empty model with the same configuration, used when decoding."""
        clone = GaussianModel(self.max_sh_degree)
        clone.active_sh_degree = self.active_sh_degree
        clone.spatial_lr_scale = getattr(self, "spatial_lr_scale", 1.0)
        clone.background = getattr(self, "background", None)
        clone.pipe = getattr(self, "pipe", None)
        if copy_qa and self.qa:
            clone.set_qa()
            for name in self.quantizer_names():
                getattr(clone, name).load_state_dict(getattr(self, name).state_dict())
        return clone

    def photometric_loss(self, image, gt, lambda_dssim):
        """Training objective of this backbone."""
        return photometric_loss(image, gt, lambda_dssim)

    def geometry_groups(self):
        """Optimizer groups frozen at the end of fine-tuning."""
        return {"xyz", "opacity", "scaling", "rotation"}

    def quantizer_names(self):
        """Stateful fake-quantize modules (the geometry ones are stateless half casts)."""
        return ["features_dc_qa", "features_rest_qa"]

    def calibrate_quantizers_for_compression(self):
        """No-op: the SH observers are updated by every quantization-aware
        forward pass, so their scales already match the tensors being encoded."""

    def train(self):
        pass

    def eval(self):
        pass

    def render(self, viewpoint_camera, mask=None, measure_fps=False):
        # Imported here because gaussian_renderer imports this module.
        from gaussian_renderer import render as render_scene
        return render_scene(viewpoint_camera, self, self.pipe, self.background,
                            mask=mask, measure_fps=measure_fps)

    def appearance_regularization(self, mask=None, weight=0.0):
        """Backbone-specific appearance prior, applied to the sampled level.
        Shrinking the higher SH bands lets them compress away."""
        if weight <= 0:
            return 0.0
        features = self._features_rest if mask is None else self._features_rest[mask]
        return weight * features.abs().mean()

    def _get_gradient_scores(self):
        """RMS-normalized per-primitive gradient importance, on CPU."""
        return self.compute_gradient_importance(score_mode="normalized").cpu()

    # ------------------------------------------------------------------
    # Compression payloads (container format: gode/streams.py)
    # ------------------------------------------------------------------

    @staticmethod
    def _quantize(tensor, qa):
        return torch.quantize_per_tensor(tensor, qa.scale, qa.zero_point, qa.dtype).int_repr().cpu().numpy()

    @staticmethod
    def _dequantize(payload, key, device):
        q = torch.from_numpy(np.ascontiguousarray(payload[key])).to(torch.int32).to(device)
        return (q - payload[f"{key}_zp"]).float() * payload[f"{key}_scale"]

    @staticmethod
    def _fp16(payload, key, device):
        return torch.from_numpy(np.ascontiguousarray(payload[key])).float().to(device)

    @torch.no_grad()
    def encode_payload(self, quantize=True):
        """Serialise the primitives into a {"geom": ..., "features": ...} payload.

        Geometry is always float16; the SH coefficients are int8 under
        quantization-aware training and float16 for the no-QAT ablation.
        """
        geom = {
            "quantized": quantize,
            "xyz":      self._xyz.detach().half().cpu().numpy(),
            "opacity":  self._opacity.detach().half().cpu().numpy(),
            "scaling":  self._scaling.detach().half().cpu().numpy(),
            "rotation": self._rotation.detach().half().cpu().numpy(),
        }
        features = {"quantized": quantize, "sh_degree": self.active_sh_degree}
        if quantize:
            features.update(
                features_dc=self._quantize(self._features_dc.detach(), self.features_dc_qa),
                features_dc_scale=float(self.features_dc_qa.scale),
                features_dc_zp=int(self.features_dc_qa.zero_point),
                features_rest=self._quantize(self._features_rest.detach(), self.features_rest_qa),
                features_rest_scale=float(self.features_rest_qa.scale),
                features_rest_zp=int(self.features_rest_qa.zero_point),
            )
        else:
            features.update(
                features_dc=self._features_dc.detach().half().cpu().numpy(),
                features_rest=self._features_rest.detach().half().cpu().numpy(),
            )
        return {"geom": geom, "features": features}

    @torch.no_grad()
    def decode_payload(self, payload):
        """Inverse of `encode_payload`; replaces the primitives in place."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        geom, features = payload["geom"], payload["features"]

        self._xyz      = self._fp16(geom, "xyz", device)
        self._opacity  = self._fp16(geom, "opacity", device)
        self._scaling  = self._fp16(geom, "scaling", device)
        self._rotation = self._fp16(geom, "rotation", device)

        if features["quantized"]:
            features_dc = self._dequantize(features, "features_dc", device)
            features_rest = self._dequantize(features, "features_rest", device)
        else:
            features_dc = self._fp16(features, "features_dc", device)
            features_rest = self._fp16(features, "features_rest", device)
        self._features_dc = nn.Parameter(features_dc, requires_grad=True)
        self._features_rest = nn.Parameter(features_rest, requires_grad=True)
        self.active_sh_degree = int(features.get("sh_degree", self.max_sh_degree))
        return True

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
        element = max(plydata.elements, key=lambda e: e.count)

        xyz = np.stack((np.asarray(element["x"]),
                        np.asarray(element["y"]),
                        np.asarray(element["z"])),  axis=1)
        opacities = np.asarray(element["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(element["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(element["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(element["f_dc_2"])

        extra_f_names = [p.name for p in element.properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(element[attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in element.properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(element[attr_name])

        rot_names = [p.name for p in element.properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(element[attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree        
        

    @torch.no_grad()
    def _sort_morton(self):
        extent = self._xyz.max(0).values - self._xyz.min(0).values
        xyz_q = ((2**21 - 1) * (self._xyz - self._xyz.min(0).values) / extent).long()
        order = mortonEncode(xyz_q).sort().indices
        self.index_primitives(order, requires_grad=False)
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