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
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from sklearn.neighbors import NearestNeighbors
import math
import torch.nn.functional as F
from gsplat.rendering import rasterization
import time
from spherical_voronoi import spherical_voronoi


class FakeQuantizationHalf(torch.autograd.Function):
    @staticmethod
    def forward(_, x):
        return x.half().float()

    @staticmethod
    def backward(_, grad_output):
        return grad_output


def _oct_encode(v: torch.Tensor) -> torch.Tensor:
    """Unit vectors [..., 3] → octahedral coords [..., 2] in [-1, 1]."""
    l1 = v.abs().sum(dim=-1, keepdim=True).clamp(min=1e-8)
    p = v[..., :2] / l1
    neg_z = v[..., 2:3] < 0
    sign_p = torch.where(p >= 0, torch.ones_like(p), -torch.ones_like(p))
    p_fold = (1.0 - p.abs().flip(-1)) * sign_p
    return torch.where(neg_z.expand_as(p), p_fold, p)


def _oct_decode(p: torch.Tensor) -> torch.Tensor:
    """Octahedral coords [..., 2] → unit vectors [..., 3]."""
    z = 1.0 - p.abs().sum(dim=-1, keepdim=True)
    sign_p = torch.where(p >= 0, torch.ones_like(p), -torch.ones_like(p))
    xy = torch.where((z < 0).expand_as(p), (1.0 - p.abs().flip(-1)) * sign_p, p)
    return F.normalize(torch.cat([xy, z], dim=-1), dim=-1)


def _splitBy3(a):
    x = a & 0x1FFFFF
    x = (x | x << 32) & 0x1F00000000FFFF
    x = (x | x << 16) & 0x1F0000FF0000FF
    x = (x | x << 8)  & 0x100F00F00F00F00F
    x = (x | x << 4)  & 0x10C30C30C30C30C3
    x = (x | x << 2)  & 0x1249249249249249
    return x


def mortonEncode(pos: torch.Tensor) -> torch.Tensor:
    x, y, z = pos.unbind(-1)
    out = torch.zeros(len(pos), dtype=torch.long, device=pos.device)
    out |= _splitBy3(x) | _splitBy3(y) << 1 | _splitBy3(z) << 2
    return out


def _quat_encode_smallest3(q: np.ndarray):
    """[N,4] unit quaternions → (small3 [N,3], idx [N] uint8).
    Drops the largest-magnitude component (reconstructable via unit norm).
    Flips sign so the dropped component is always positive."""
    N = len(q)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
    idx = np.abs(q).argmax(axis=-1)
    signs = np.where(q[np.arange(N), idx] >= 0, 1.0, -1.0)
    q = q * signs[:, None]
    mask = np.ones((N, 4), dtype=bool)
    mask[np.arange(N), idx] = False
    return q[mask].reshape(N, 3).astype(np.float32), idx.astype(np.uint8)


def _quat_decode_smallest3(small3: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Inverse of _quat_encode_smallest3. Returns [N,4] float32."""
    N = len(idx)
    q = np.zeros((N, 4), dtype=np.float32)
    mask = np.ones((N, 4), dtype=bool)
    mask[np.arange(N), idx] = False
    q[mask] = small3.reshape(-1)
    q[np.arange(N), idx] = np.sqrt(np.maximum(1.0 - (small3 ** 2).sum(axis=-1), 0.0))
    return q


def _pack_rot_idx(idx: np.ndarray) -> np.ndarray:
    """[N] uint8 in {0-3} → packed [ceil(N/4)] uint8, 2 bits per entry."""
    pad = (-len(idx)) % 4
    p = np.concatenate([idx, np.zeros(pad, dtype=np.uint8)])
    return (p[0::4] | (p[1::4] << 2) | (p[2::4] << 4) | (p[3::4] << 6)).astype(np.uint8)


def _unpack_rot_idx(packed: np.ndarray, N: int) -> np.ndarray:
    """packed [ceil(N/4)] uint8 → [N] uint8 in {0-3}."""
    out = np.empty(len(packed) * 4, dtype=np.uint8)
    out[0::4] = packed & 0x03
    out[1::4] = (packed >> 2) & 0x03
    out[2::4] = (packed >> 4) & 0x03
    out[3::4] = (packed >> 6) & 0x03
    return out[:N]


# raw init so that F.softplus(x) == 1.0
_SOFTPLUS_INV_1 = float(torch.log(torch.exp(torch.tensor(1.0)) - 1.0))


def knn(x, K=4):
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


def fibonacci_sphere(N):
    indices = torch.arange(0, N, dtype=torch.float32) + 0.5
    phi = math.pi * (3. - math.sqrt(5.))
    y = 1 - (indices / N) * 2
    radius = torch.sqrt(1 - y ** 2)
    theta = phi * indices
    x = torch.cos(theta) * radius
    z = torch.sin(theta) * radius
    return torch.stack([x, y, z], dim=1)


class BetaModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        def sb_params_activation(sb_params):
            softplus_sb_params = F.softplus(sb_params[..., :3], beta=math.log(2) * 10)
            sb_params = torch.cat([softplus_sb_params, sb_params[..., 3:]], dim=-1)
            return sb_params

        def beta_activation(betas):
            return 4.0 * torch.exp(betas)

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        self.sb_params_activation = sb_params_activation
        self.beta_activation = beta_activation
        self.sites_activation = torch.nn.functional.normalize

    def __init__(self, color_representation='sh', num_lobes=8):
        """
        Args:
            color_representation: 'voronoi' | 'sh' | 'sb' | 'sgs'
            num_lobes: Voronoi / SGS lobes (ignored for sh / sb)
        """
        self.active_sh_degree = 0
        self.max_sh_degree = 3 if color_representation == 'sh' else 0
        self.sb_number = 2 if color_representation == 'sb' else 0
        self._xyz = torch.empty(0)
        self._sh0 = torch.empty(0)
        self._shN = torch.empty(0)
        self._sb_params = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._beta = torch.empty(0)
        self.background = torch.empty(0)
        self.training = True
        self.optimizer = None
        self.spatial_lr_scale = 0
        self.color_representation = color_representation
        self.qa = False

        if color_representation == 'voronoi':
            self._sites = torch.empty(0)
            self._colors = torch.empty(0)
            self._log_tau_sites = torch.empty(0)
            self.num_lobes = num_lobes

        if color_representation == 'sgs':
            self._mu = torch.empty(0)
            self._alpha = torch.empty(0)
            self._sharpness = torch.empty(0)
            self._dc = torch.empty(0)
            self.num_lobes = num_lobes

        self.setup_functions()

    # ------------------------------------------------------------------
    # Primitive bookkeeping
    # The scalable core only touches primitives through these methods, which
    # is what keeps it backbone-agnostic.
    # ------------------------------------------------------------------

    @property
    def num_primitives(self):
        return self._xyz.shape[0]

    def geometry_groups(self):
        """Optimizer groups frozen at the end of the scalable fine-tuning."""
        return {"xyz", "opacity", "beta", "scaling", "rotation"}

    def photometric_loss(self, image, gt, lambda_dssim):
        """Training objective of this backbone."""
        return photometric_loss(image, gt, lambda_dssim)

    def appearance_regularization(self, mask=None, weight=0.0):
        """Backbone-specific appearance prior, applied to the sampled level.
        Pulling the Voronoi colors toward black lets unused lobes compress away."""
        if weight <= 0 or self.color_representation != "voronoi":
            return 0.0
        colors = self._colors if mask is None else self._colors[mask]
        return weight * (colors + 0.5).abs().mean()

    def primitive_names(self, include_empty=False):
        """Attributes holding one row per primitive, in a stable order."""
        names = ["_xyz", "_opacity", "_beta", "_scaling", "_rotation",
                 "_sh0", "_shN", "_sb_params"]
        if self.color_representation == "voronoi":
            names += ["_sites", "_colors", "_log_tau_sites"]
        elif self.color_representation == "sgs":
            names += ["_mu", "_alpha", "_sharpness", "_dc"]
        if include_empty:
            return names
        return [n for n in names if getattr(self, n, torch.empty(0)).numel() > 0]

    def _optimizable_dict(self):
        """Primitives keyed by optimizer parameter-group name."""
        return {n[1:]: getattr(self, n) for n in self.primitive_names(include_empty=True)}

    def _assign_optimizable(self, tensors):
        for name, tensor in tensors.items():
            setattr(self, f"_{name}", tensor)

    def index_primitives(self, index, requires_grad=True):
        """Select or reorder primitives along dim 0 (boolean mask or indices)."""
        for name in self.primitive_names():
            tensor = getattr(self, name)
            tensor = tensor.data if isinstance(tensor, nn.Parameter) else tensor
            value = tensor[index.to(tensor.device)].clone()
            setattr(self, name, nn.Parameter(value, requires_grad=requires_grad))

    def subset(self, mask, copy_qa=True):
        """A detached copy holding only the selected primitives."""
        clone = self.clone_empty(copy_qa=copy_qa)
        for name in self.primitive_names():
            tensor = getattr(self, name)
            tensor = tensor.data if isinstance(tensor, nn.Parameter) else tensor
            setattr(clone, name, nn.Parameter(tensor[mask].clone(), requires_grad=False))
        return clone

    def append_primitives(self, other):
        """Append another model's primitives (incremental LoD decoding)."""
        for name in self.primitive_names():
            mine, theirs = getattr(self, name), getattr(other, name)
            mine = mine.data if isinstance(mine, nn.Parameter) else mine
            theirs = theirs.data if isinstance(theirs, nn.Parameter) else theirs
            setattr(self, name, torch.cat([mine, theirs.to(mine.device)], dim=0))

    def clone_empty(self, copy_qa=False):
        """An empty model with the same configuration, used when decoding."""
        clone = BetaModel(self.color_representation, num_lobes=getattr(self, "num_lobes", 8))
        clone.background = self.background
        clone.max_sh_degree = self.max_sh_degree
        clone.active_sh_degree = self.active_sh_degree
        clone.sb_number = self.sb_number
        clone.spatial_lr_scale = self.spatial_lr_scale
        if copy_qa and self.qa:
            clone.set_qa()
            for name in self.quantizer_names():
                getattr(clone, name).load_state_dict(getattr(self, name).state_dict())
        return clone

    def quantizer_names(self):
        """Stateful fake-quantize modules (the plain functional ones are stateless)."""
        names = ["opacity_qa", "rotation_qa", "beta_qa"]
        if self.color_representation == "voronoi":
            return names + ["sites_qa", "colors_qa", "log_tau_qa"]
        return names + ["sh0_qa", "shN_qa"]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def get_scaling(self):
        s = self.scaling_qa(self._scaling) if self.qa else self._scaling
        return self.scaling_activation(s)

    @property
    def get_rotation(self):
        if self.qa:
            return self.rotation_activation(self.rotation_qa(self._rotation))
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self.xyz_qa(self._xyz) if self.qa else self._xyz

    @property
    def get_shs(self):
        if self.qa:
            return torch.cat((self.sh0_qa(self._sh0), self.shN_qa(self._shN)), dim=1)
        return torch.cat((self._sh0, self._shN), dim=1)

    @property
    def get_beta(self):
        b = self.beta_qa(self._beta) if self.qa else self._beta
        return self.beta_activation(b)

    @property
    def get_sb_params(self):
        return self.sb_params_activation(self._sb_params)

    @property
    def get_opacity(self):
        o = self.opacity_activation(self._opacity)
        return self.opacity_qa(o) if self.qa else o

    @property
    def get_mu(self):
        return F.normalize(self._mu, dim=-1)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    # ------------------------------------------------------------------
    # Quantization-aware training setup
    # ------------------------------------------------------------------

    def set_qa(self):
        self.qa = True
        self.xyz_qa      = FakeQuantizationHalf.apply
        self.beta_qa     = torch.ao.quantization.FakeQuantize(dtype=torch.qint8).cuda()
        self.scaling_qa  = FakeQuantizationHalf.apply
        self.opacity_qa  = torch.ao.quantization.FakeQuantize(dtype=torch.quint8).cuda()
        self.rotation_qa = torch.ao.quantization.FakeQuantize(dtype=torch.qint8).cuda()
        self.sh0_qa = torch.ao.quantization.FakeQuantize(dtype=torch.qint8).cuda()
        self.shN_qa = torch.ao.quantization.FakeQuantize(dtype=torch.qint8).cuda()
        if self.color_representation == 'voronoi':
            self.sites_qa   = torch.ao.quantization.FakeQuantize(dtype=torch.qint8).cuda()
            self.colors_qa  = torch.ao.quantization.FakeQuantize(dtype=torch.qint8).cuda()
            self.log_tau_qa = torch.ao.quantization.FakeQuantize(dtype=torch.qint8).cuda()

    @torch.no_grad()
    def calibrate_quantizers_for_compression(self):
        if not getattr(self, 'qa', False):
            return
        # Geometry
        self.opacity_qa(self.opacity_activation(self._opacity.detach()).reshape(-1))
        self.rotation_qa(self._rotation.detach().reshape(-1))
        self.beta_qa(self._beta.detach().reshape(-1))
        print(
            f"[QA-CAL] opacity  scale={self.opacity_qa.scale.item():.6g} zp={int(self.opacity_qa.zero_point.item())}  "
            f"rotation scale={self.rotation_qa.scale.item():.6g} zp={int(self.rotation_qa.zero_point.item())}  "
            f"beta     scale={self.beta_qa.scale.item():.6g} zp={int(self.beta_qa.zero_point.item())}"
        )
        if self.color_representation == 'voronoi':
            if self._colors.numel() == 0:
                return
            sites_oct = _oct_encode(F.normalize(self._sites.detach(), dim=-1)).reshape(-1, 2)
            colors    = self._colors.detach().reshape(-1, 3)
            log_tau   = self._log_tau_sites.detach().reshape(-1)
            self.sites_qa(sites_oct)
            self.colors_qa(colors)
            self.log_tau_qa(log_tau)
            print(
                f"[QA-CAL] sites  scale={self.sites_qa.scale.item():.6g} zp={int(self.sites_qa.zero_point.item())}  "
                f"colors scale={self.colors_qa.scale.item():.6g} zp={int(self.colors_qa.zero_point.item())}  "
                f"log_tau scale={self.log_tau_qa.scale.item():.6g} zp={int(self.log_tau_qa.zero_point.item())}"
            )
        elif self.color_representation == 'sh':
            if self._sh0.numel() > 0:
                self.sh0_qa(self._sh0.detach())
            if self._shN.numel() > 0:
                self.shN_qa(self._shN.detach())

    # ------------------------------------------------------------------
    # Train / eval mode
    # ------------------------------------------------------------------

    @torch.no_grad()
    def eval(self):
        self.training = False
        if self.color_representation == 'voronoi':
            if getattr(self, 'qa', False):
                # Simulate quantization so intermediate evals reflect decode quality
                sites_oct = _oct_encode(F.normalize(self._sites, dim=-1))
                self._nn_sites = _oct_decode(self.sites_qa(sites_oct)).detach()
                log_tau_q = self.log_tau_qa(self._log_tau_sites)
                self._nn_tau = torch.exp(log_tau_q).detach()
                self._nn_colors = self.colors_qa(self._colors).detach()
            else:
                self._nn_sites = F.normalize(self._sites, dim=-1)
                self._nn_tau   = torch.exp(self._log_tau_sites).detach()
                self._nn_colors = self._colors.detach()

    def train(self):
        self.training = True

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # ------------------------------------------------------------------
    # Color computation
    # ------------------------------------------------------------------

    def compute_voronoi_color(self, directions: torch.Tensor, mask=None) -> torch.Tensor:
        def _m(t):
            return t if mask is None else t[mask]
        if self.training:
            sites_n = F.normalize(_m(self._sites), dim=-1)
            log_tau = _m(self._log_tau_sites)
            colors  = _m(self._colors)
            if self.qa:
                sites_n = _oct_decode(self.sites_qa(_oct_encode(sites_n)))
                colors  = self.colors_qa(colors)
                log_tau = self.log_tau_qa(log_tau)
            tau = torch.exp(log_tau)
        else:
            sites_n = _m(self._nn_sites)
            tau     = _m(self._nn_tau)
            colors  = _m(self._nn_colors)
        return spherical_voronoi(directions.reshape(-1, 3), sites_n, tau, colors)

    def compute_radiance(self, directions, mask=None):
        if self.color_representation == 'voronoi':
            return self.compute_voronoi_color(directions.reshape(-1, 3), mask=mask)

        if self.color_representation == 'sgs':
            def _m(t):
                return t if mask is None else t[mask]
            directions = directions.reshape(-1, 1, 3)
            mu = F.normalize(_m(self._mu), dim=-1)
            dot_product = (mu * directions).sum(dim=-1).unsqueeze(-1)
            sharpness = F.softplus(_m(self._sharpness))
            V = _m(self._alpha) * torch.exp(sharpness * (dot_product - 1))
            V = V.sum(dim=1)
            return torch.clamp_min(V, 0.0)

    # ------------------------------------------------------------------
    # Initialisation from point cloud
    # ------------------------------------------------------------------

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        shs = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        shs[:, :3, 0] = fused_color
        shs[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = (
            knn(torch.from_numpy(np.asarray(pcd.points)).float().cuda())[:, 1:] ** 2
        ).mean(dim=-1)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(
            0.5 * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )
        betas = torch.zeros_like(opacities)

        sb_params = torch.zeros(
            (fused_point_cloud.shape[0], self.sb_number, 6), device="cuda"
        )
        theta = torch.pi * torch.rand(fused_point_cloud.shape[0], self.sb_number)
        phi = 2 * torch.pi * torch.rand(fused_point_cloud.shape[0], self.sb_number)
        sb_params[:, :, 3] = theta
        sb_params[:, :, 4] = phi

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._sh0 = nn.Parameter(
            shs[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._shN = nn.Parameter(
            shs[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._sb_params = nn.Parameter(sb_params.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._beta = nn.Parameter(betas.requires_grad_(True))

        if self.color_representation == 'voronoi':
            N = self._xyz.shape[0]
            vectors = fibonacci_sphere(self.num_lobes).repeat(N, 1, 1).to('cuda')
            self._sites = nn.Parameter(vectors)
            colors = torch.tensor(np.asarray(pcd.colors)).float().cuda() - 0.5
            self._colors = nn.Parameter(colors.repeat(1, self.num_lobes).reshape(N, -1, 3))
            self._log_tau_sites = nn.Parameter(torch.zeros(N, self.num_lobes, device='cuda'))

        if self.color_representation == 'sgs':
            N = self._xyz.shape[0]
            mu = fibonacci_sphere(self.num_lobes).repeat(N, 1, 1).to('cuda')
            self._mu = nn.Parameter(mu)
            colors = torch.tensor(np.asarray(pcd.colors)).float().cuda()
            self._alpha = nn.Parameter(
                colors.repeat(1, self.num_lobes).reshape(N, -1, 3)
            )
            self._dc = nn.Parameter(
                torch.zeros((N, 3), device='cuda', requires_grad=True)
            )
            self._sharpness = nn.Parameter(
                torch.full((N, self.num_lobes, 1), 5.0, device='cuda', requires_grad=True)
            )

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    CONFIG_KEYS = ("color_representation", "spatial_lr_scale", "active_sh_degree",
                   "sb_number", "max_sh_degree", "num_lobes")

    def save(self, path):
        torch.save({
            "tensors": {n: getattr(self, n).detach() for n in self.primitive_names()},
            "config": {k: getattr(self, k) for k in self.CONFIG_KEYS if hasattr(self, k)},
            "optimizer_state": self.optimizer.state_dict() if self.optimizer is not None else None,
        }, path)

    def restore(self, path, training_args):
        """Load a checkpoint written by save() and rebuild the optimizer."""
        data = torch.load(path, map_location="cuda")
        for key, value in data["config"].items():
            setattr(self, key, value)
        for name, value in data["tensors"].items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, nn.Parameter(value.cuda().requires_grad_(True)))
            else:
                setattr(self, name, value)   # older checkpoints stored scalars here

        self.training_setup(training_args)
        state = data.get("optimizer_state")
        if state is not None and self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(state)
            except (ValueError, KeyError):
                print("[restore] optimizer state does not match; starting a fresh optimizer")

    # ------------------------------------------------------------------
    # PLY save / load
    # ------------------------------------------------------------------

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        for i in range(self._sh0.shape[1] * self._sh0.shape[2]):
            l.append(f"sh0_{i}")
        for i in range(self._shN.shape[1] * self._shN.shape[2]):
            l.append(f"shN_{i}")
        for i in range(self._sb_params.shape[1] * self._sb_params.shape[2]):
            l.append(f"sb_params_{i}")
        l.append("opacity")
        l.append("beta")
        for i in range(self._scaling.shape[1]):
            l.append(f"scale_{i}")
        for i in range(self._rotation.shape[1]):
            l.append(f"rot_{i}")

        if self.color_representation == "voronoi" and hasattr(self, "_sites") and self._sites.numel() > 0:
            L = self._sites.shape[1]
            for li in range(L):
                l += [f"site_{li}_x", f"site_{li}_y", f"site_{li}_z",
                      f"color_{li}_r", f"color_{li}_g", f"color_{li}_b",
                      f"tau_{li}"]

        if self.color_representation == "sgs" and hasattr(self, "_mu") and self._mu.numel() > 0:
            L = self._mu.shape[1]
            l += ["dc_r", "dc_g", "dc_b"]
            for li in range(L):
                l += [f"mu_{li}_x", f"mu_{li}_y", f"mu_{li}_z",
                      f"alpha_{li}_r", f"alpha_{li}_g", f"alpha_{li}_b",
                      f"sharp_{li}"]
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        xyz      = self._xyz.detach().cpu().numpy()
        normals  = np.zeros_like(xyz)
        sh0      = self._sh0.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        shN      = self._shN.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        if self._sb_params.numel() > 0:
            sb_params = self._sb_params.transpose(1, 2).detach().flatten(start_dim=1).contiguous().cpu().numpy()
        else:
            sb_params = np.zeros((xyz.shape[0], 0), dtype=np.float32)
        opacities = self._opacity.detach().cpu().numpy()
        betas     = self._beta.detach().cpu().numpy()
        scale     = self._scaling.detach().cpu().numpy()
        rotation  = self._rotation.detach().cpu().numpy()

        extra_cols = []

        if self.color_representation == "voronoi" and hasattr(self, "_sites") and self._sites.numel() > 0:
            sites   = self._sites.detach().cpu().numpy()
            colors  = self._colors.detach().cpu().numpy()
            log_tau = self._log_tau_sites.detach().cpu().numpy()  # [N, L]
            N, L, _ = sites.shape
            for li in range(L):
                extra_cols.append(sites[:, li, :])
                extra_cols.append(colors[:, li, :])
                extra_cols.append(log_tau[:, li: li + 1])

        if self.color_representation == "sgs" and hasattr(self, "_mu") and self._mu.numel() > 0:
            mu    = self._mu.detach().cpu().numpy()
            alpha = self._alpha.detach().cpu().numpy()
            sharp = self._sharpness.detach().cpu().numpy()
            dc    = self._dc.detach().cpu().numpy()
            N, L, _ = mu.shape
            extra_cols.append(dc)
            for li in range(L):
                extra_cols.append(mu[:, li, :])
                extra_cols.append(alpha[:, li, :])
                extra_cols.append(sharp[:, li, :])

        base_attrs = [xyz, normals, sh0, shN, sb_params, opacities, betas, scale, rotation]
        if len(extra_cols) > 0:
            attributes = np.concatenate(base_attrs + extra_cols, axis=1)
        else:
            attributes = np.concatenate(base_attrs, axis=1)

        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]
        assert attributes.shape[1] == len(dtype_full), \
            f"Mismatch: attributes {attributes.shape[1]} vs dtype {len(dtype_full)}"
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata  = PlyData.read(path)
        elem     = plydata.elements[0]
        xyz      = np.stack((np.asarray(elem["x"]), np.asarray(elem["y"]), np.asarray(elem["z"])), axis=1)
        opacities = np.asarray(elem["opacity"])[..., np.newaxis]
        betas    = np.asarray(elem["beta"])[..., np.newaxis]

        sh0 = np.zeros((xyz.shape[0], 3, 1))
        sh0[:, 0, 0] = np.asarray(elem["sh0_0"])
        sh0[:, 1, 0] = np.asarray(elem["sh0_1"])
        sh0[:, 2, 0] = np.asarray(elem["sh0_2"])

        prop_names = [p.name for p in elem.properties]
        shN_names = sorted(
            [n for n in prop_names if n.startswith("shN_")],
            key=lambda x: int(x.split("_")[-1])
        )
        assert len(shN_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        shs_extra = np.zeros((xyz.shape[0], len(shN_names)))
        for idx, attr_name in enumerate(shN_names):
            shs_extra[:, idx] = np.asarray(elem[attr_name])
        shs_extra = shs_extra.reshape(
            (shs_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        sb_names = sorted(
            [n for n in prop_names if n.startswith("sb_params_")],
            key=lambda x: int(x.split("_")[-1])
        )
        if self.sb_number > 0:
            assert len(sb_names) == self.sb_number * 6
            sb_params = np.zeros((xyz.shape[0], len(sb_names)))
            for idx, attr_name in enumerate(sb_names):
                sb_params[:, idx] = np.asarray(elem[attr_name])
            sb_params = sb_params.reshape((sb_params.shape[0], 6, self.sb_number))
        else:
            sb_params = np.zeros((xyz.shape[0], 6, 0))

        scale_names = sorted(
            [n for n in prop_names if n.startswith("scale_")],
            key=lambda x: int(x.split("_")[-1])
        )
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(elem[attr_name])

        rot_names = sorted(
            [n for n in prop_names if n.startswith("rot_")],
            key=lambda x: int(x.split("_")[-1])
        )
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(elem[attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._sh0 = nn.Parameter(
            torch.tensor(sh0, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._shN = nn.Parameter(
            torch.tensor(shs_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._sb_params = nn.Parameter(
            torch.tensor(sb_params, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._beta = nn.Parameter(
            torch.tensor(betas, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self.active_sh_degree = self.max_sh_degree

        # --- voronoi ---
        if self.color_representation == "voronoi":
            site_xyz_names = [
                n for n in prop_names
                if n.startswith("site_") and (n.endswith("_x") or n.endswith("_y") or n.endswith("_z"))
            ]
            color_names = [
                n for n in prop_names
                if n.startswith("color_") and (n.endswith("_r") or n.endswith("_g") or n.endswith("_b"))
            ]
            tau_names = sorted(
                [n for n in prop_names if n.startswith("tau_")],
                key=lambda x: int(x.split("_")[1])
            )
            if len(site_xyz_names) > 0:
                def site_key(name):
                    parts = name.split("_")
                    return int(parts[1]) * 3 + {"x": 0, "y": 1, "z": 2}[parts[2]]

                def color_key(name):
                    parts = name.split("_")
                    return int(parts[1]) * 3 + {"r": 0, "g": 1, "b": 2}[parts[2]]

                site_xyz_names = sorted(site_xyz_names, key=site_key)
                color_names    = sorted(color_names, key=color_key)

                sites_flat = np.zeros((xyz.shape[0], len(site_xyz_names)))
                for idx, attr_name in enumerate(site_xyz_names):
                    sites_flat[:, idx] = np.asarray(elem[attr_name])
                L = len(site_xyz_names) // 3
                sites = sites_flat.reshape(xyz.shape[0], L, 3)

                colors_flat = np.zeros((xyz.shape[0], len(color_names)))
                for idx, attr_name in enumerate(color_names):
                    colors_flat[:, idx] = np.asarray(elem[attr_name])
                colors = colors_flat.reshape(xyz.shape[0], L, 3)

                if len(tau_names) > 0:
                    tau_flat = np.zeros((xyz.shape[0], len(tau_names)))
                    for idx, attr_name in enumerate(tau_names):
                        tau_flat[:, idx] = np.asarray(elem[attr_name])
                else:
                    tau_flat = np.zeros((xyz.shape[0], L), dtype=np.float32)

                self._sites = nn.Parameter(
                    torch.tensor(sites, dtype=torch.float, device="cuda").requires_grad_(True)
                )
                self._colors = nn.Parameter(
                    torch.tensor(colors, dtype=torch.float, device="cuda").requires_grad_(True)
                )
                self._log_tau_sites = nn.Parameter(
                    torch.tensor(tau_flat, dtype=torch.float, device="cuda").requires_grad_(True)
                )
                self.num_lobes = L

        # --- sgs ---
        if self.color_representation == "sgs":
            dc = None
            if all(n in prop_names for n in ["dc_r", "dc_g", "dc_b"]):
                dc = np.stack(
                    (np.asarray(elem["dc_r"]), np.asarray(elem["dc_g"]), np.asarray(elem["dc_b"])),
                    axis=1,
                )
            mu_names    = [n for n in prop_names if n.startswith("mu_") and (n.endswith("_x") or n.endswith("_y") or n.endswith("_z"))]
            alpha_names = [n for n in prop_names if n.startswith("alpha_") and (n.endswith("_r") or n.endswith("_g") or n.endswith("_b"))]
            sharp_names = [n for n in prop_names if n.startswith("sharp_")]
            if len(mu_names) > 0:
                def mu_key(name):
                    parts = name.split("_")
                    return int(parts[1]) * 3 + {"x": 0, "y": 1, "z": 2}[parts[2]]

                def alpha_key(name):
                    parts = name.split("_")
                    return int(parts[1]) * 3 + {"r": 0, "g": 1, "b": 2}[parts[2]]

                mu_names    = sorted(mu_names, key=mu_key)
                alpha_names = sorted(alpha_names, key=alpha_key)
                sharp_names = sorted(sharp_names, key=lambda x: int(x.split("_")[1]))

                mu_flat = np.zeros((xyz.shape[0], len(mu_names)))
                for idx, attr_name in enumerate(mu_names):
                    mu_flat[:, idx] = np.asarray(elem[attr_name])
                L = len(mu_names) // 3
                mu = mu_flat.reshape(xyz.shape[0], L, 3)

                alpha_flat = np.zeros((xyz.shape[0], len(alpha_names)))
                for idx, attr_name in enumerate(alpha_names):
                    alpha_flat[:, idx] = np.asarray(elem[attr_name])
                alpha = alpha_flat.reshape(xyz.shape[0], L, 3)

                sharp_flat = np.zeros((xyz.shape[0], len(sharp_names)))
                for idx, attr_name in enumerate(sharp_names):
                    sharp_flat[:, idx] = np.asarray(elem[attr_name])
                sharp = sharp_flat.reshape(xyz.shape[0], L, 1)

                self._mu = nn.Parameter(
                    torch.tensor(mu, dtype=torch.float, device="cuda").requires_grad_(True)
                )
                self._alpha = nn.Parameter(
                    torch.tensor(alpha, dtype=torch.float, device="cuda").requires_grad_(True)
                )
                self._sharpness = nn.Parameter(
                    torch.tensor(sharp, dtype=torch.float, device="cuda").requires_grad_(True)
                )
                if dc is not None:
                    self._dc = nn.Parameter(
                        torch.tensor(dc, dtype=torch.float, device="cuda").requires_grad_(True)
                    )
                self.num_lobes = L

    # ------------------------------------------------------------------
    # Optimizer setup
    # ------------------------------------------------------------------

    def training_setup(self, training_args):
        l = [
            {"params": [self._xyz],      "lr": training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {"params": [self._opacity],  "lr": training_args.opacity_lr,  "name": "opacity"},
            {"params": [self._beta],     "lr": training_args.beta_lr,     "name": "beta"},
            {"params": [self._scaling],  "lr": training_args.scaling_lr,  "name": "scaling"},
            {"params": [self._rotation], "lr": training_args.rotation_lr, "name": "rotation"},
        ]

        if self.color_representation in ("sh", "sb"):
            l += [
                {"params": [self._sh0],       "lr": training_args.sh_lr,            "name": "sh0"},
                {"params": [self._shN],       "lr": training_args.sh_lr / 20.0,     "name": "shN"},
                {"params": [self._sb_params], "lr": training_args.sb_params_lr,     "name": "sb_params"},
            ]

        elif self.color_representation == "voronoi":
            sites_lr = training_args.sites_lr
            color_lr = training_args.color_lr
            tau_lr   = getattr(training_args, "tau_lr", sites_lr)
            self.voronoi_sites_lr_init = sites_lr
            self.voronoi_tau_lr_init   = tau_lr
            self.total_iterations = getattr(training_args, "iterations", 45000)
            l += [
                {"params": [self._sh0],            "lr": training_args.sh_lr,        "name": "sh0"},
                {"params": [self._shN],            "lr": training_args.sh_lr / 20.0, "name": "shN"},
                {"params": [self._sb_params],      "lr": training_args.sb_params_lr, "name": "sb_params"},
                {"params": [self._sites],          "lr": sites_lr,                   "name": "sites"},
                {"params": [self._colors],         "lr": color_lr,                   "name": "colors"},
                {"params": [self._log_tau_sites],  "lr": tau_lr,                     "name": "log_tau_sites"},
            ]

        elif self.color_representation == "sgs":
            l += [
                {"params": [self._sh0],       "lr": training_args.sh_lr,        "name": "sh0"},
                {"params": [self._shN],       "lr": training_args.sh_lr / 20.0, "name": "shN"},
                {"params": [self._sb_params], "lr": training_args.sb_params_lr, "name": "sb_params"},
                {"params": [self._mu],        "lr": training_args.sites_lr,     "name": "mu"},
                {"params": [self._alpha],     "lr": training_args.color_lr,     "name": "alpha"},
                {"params": [self._sharpness], "lr": 1e-2,                       "name": "sharpness"},
            ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )
        self.sites_scheduler_args = get_expon_lr_func(
            lr_init=5e-2, lr_final=1e-4,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=30_000,
        )

    def update_learning_rate(self, iteration):
        xyz_lr = None
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                xyz_lr = param_group["lr"]

        if self.color_representation == "voronoi" and hasattr(self, "voronoi_sites_lr_init"):
            T = self.total_iterations
            t = math.pi * iteration / T

            lr0_sites = self.voronoi_sites_lr_init
            cos_lr_sites = lr0_sites * 0.1 + 0.5 * lr0_sites * 0.9 * (1 + math.cos(t))

            lr0_tau = self.voronoi_tau_lr_init
            cos_lr_tau = lr0_tau * 0.1 + 0.5 * lr0_tau * 0.9 * (1 + math.cos(t))

            for pg in self.optimizer.param_groups:
                if pg["name"] == "sites":
                    pg["lr"] = cos_lr_sites
                elif pg["name"] == "log_tau_sites":
                    pg["lr"] = cos_lr_tau
        return xyz_lr

    # ------------------------------------------------------------------
    # Optimizer helpers
    # ------------------------------------------------------------------

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"]    = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"]    = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0
                )
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def densification_postfix(self, params, reset_params=True):
        optimizable_tensors = self.cat_tensors_to_optimizer(params)
        self._assign_optimizable(optimizable_tensors)

    def replace_tensors_to_optimizer(self, inds=None):
        tensors_dict = self._optimizable_dict()

        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            tensor = tensors_dict[group["name"]]
            if tensor.numel() == 0:
                optimizable_tensors[group["name"]] = group["params"][0]
                continue
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                if inds is not None:
                    stored_state["exp_avg"][inds]    = 0
                    stored_state["exp_avg_sq"][inds] = 0
                else:
                    stored_state["exp_avg"]    = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state
            optimizable_tensors[group["name"]] = group["params"][0]

        self._assign_optimizable(optimizable_tensors)

        torch.cuda.empty_cache()
        return optimizable_tensors

    # ------------------------------------------------------------------
    # Densification
    # ------------------------------------------------------------------

    def _update_params(self, idxs, ratio):
        new_opacity = 1.0 - torch.pow(1.0 - self.get_opacity[idxs, 0], 1.0 / (ratio + 1))
        new_opacity = torch.clamp(
            new_opacity.unsqueeze(-1),
            max=1.0 - torch.finfo(torch.float32).eps, min=0.005
        )
        new_opacity = self.inverse_opacity_activation(new_opacity)
        ret = [
            self._xyz[idxs], self._sh0[idxs], self._shN[idxs],
            self._sb_params[idxs], new_opacity, self._beta[idxs],
            self._scaling[idxs], self._rotation[idxs],
        ]
        if self.color_representation == "voronoi":
            ret += [self._sites[idxs], self._colors[idxs], self._log_tau_sites[idxs]]
        elif self.color_representation == "sgs":
            ret += [self._mu[idxs], self._alpha[idxs], self._sharpness[idxs]]
        return (*ret,)

    def _sample_alives(self, probs, num, alive_indices=None):
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
        sampled_idxs = torch.multinomial(probs, num, replacement=True)
        if alive_indices is not None:
            sampled_idxs = alive_indices[sampled_idxs]
        ratio = torch.bincount(sampled_idxs)[sampled_idxs]
        return sampled_idxs, ratio

    def relocate_gs(self, dead_mask=None):
        if dead_mask.sum() == 0:
            return
        alive_mask    = ~dead_mask
        dead_indices  = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]
        if alive_indices.shape[0] <= 0:
            return
        probs = self.get_opacity[alive_indices, 0]
        reinit_idx, ratio = self._sample_alives(
            alive_indices=alive_indices, probs=probs, num=dead_indices.shape[0]
        )
        (self._xyz[dead_indices], self._sh0[dead_indices], self._shN[dead_indices],
         self._sb_params[dead_indices], self._opacity[dead_indices], self._beta[dead_indices],
         self._scaling[dead_indices], self._rotation[dead_indices], *rest) = \
            self._update_params(reinit_idx, ratio=ratio)

        if self.color_representation == "voronoi":
            self._sites[dead_indices]         = rest[0]
            self._colors[dead_indices]        = rest[1]
            self._log_tau_sites[dead_indices] = rest[2]
        elif self.color_representation == "sgs":
            self._mu[dead_indices]        = rest[0]
            self._alpha[dead_indices]     = rest[1]
            self._sharpness[dead_indices] = rest[2]

        self._opacity[reinit_idx] = self._opacity[dead_indices]
        self.replace_tensors_to_optimizer(inds=reinit_idx)

    def subsample_to(self, cap_max):
        N = self._xyz.shape[0]
        if N <= cap_max:
            return
        perm = torch.randperm(N, device='cuda')[:cap_max]
        mask = torch.zeros(N, dtype=torch.bool, device='cuda')
        mask[perm] = True

        optimizable_tensors = self._prune_optimizer(mask)
        self._assign_optimizable(optimizable_tensors)


    # ------------------------------------------------------------------
    # Gradient-based pruning for hierarchy construction
    # ------------------------------------------------------------------

    def _get_gradient_score_components(self):
        """Return per-Gaussian gradient importance score, split by parameter group
        (N,) tensors on the model's device. Used for the current combined score
        (see _get_gradient_scores) and for ablating which groups matter."""
        def _safe_norm(t, *dims):
            if t is None or t.grad is None:
                return 0.0
            g = t.grad
            for d in dims:
                g = g.norm(dim=d)
            return g

        comps = {
            "xyz":      _safe_norm(self._xyz, 1),
            "opacity":  _safe_norm(self._opacity, 1),
            "beta":     _safe_norm(self._beta, 1),
            "scaling":  _safe_norm(self._scaling, 1),
            "rotation": _safe_norm(self._rotation, 1),
        }
        color = _safe_norm(self._sh0, -1, -1) + _safe_norm(self._shN, -1, -1)
        if self.color_representation == 'voronoi':
            color = (
                _safe_norm(self._sites, -1, -1) +
                _safe_norm(self._colors, -1, -1) +
                _safe_norm(self._log_tau_sites, -1)
            )
        elif self.color_representation == 'sgs':
            color = (
                _safe_norm(self._mu, -1, -1) +
                _safe_norm(self._alpha, -1, -1) +
                _safe_norm(self._sharpness, -1, -1)
            )
        comps["color"] = color

        n = self._xyz.shape[0]
        for k, v in comps.items():
            if isinstance(v, float):
                comps[k] = torch.zeros(n, device=self._xyz.device)
        return comps

    def _get_gradient_scores(self):
        """Return RMS-normalized per-Gaussian gradient importance scores on CPU."""
        components = self._get_gradient_score_components()
        normalized = []
        num_gaussians = self._xyz.shape[0]

        for name, contribution in components.items():
            if contribution.ndim != 1 or contribution.shape[0] != num_gaussians:
                raise RuntimeError(
                    f"Invalid gradient score shape for {name}: "
                    f"expected ({num_gaussians},), got {tuple(contribution.shape)}"
                )
            if not torch.isfinite(contribution).all():
                raise RuntimeError(f"Non-finite gradient contribution for {name}.")

            epsilon = torch.finfo(contribution.dtype).eps
            rms = contribution.square().mean().sqrt().clamp_min(epsilon)
            normalized.append(contribution / rms)

        score = torch.stack(normalized, dim=0).sum(dim=0).detach()
        if not torch.isfinite(score).all():
            raise RuntimeError("Non-finite normalized gradient importance score.")
        return score.cpu()

    def gradient_prune(self, k, prune_type='gradient_all', largest=False, scores=None):
        if prune_type == 'random':
            indices = torch.multinomial(
                torch.ones(self._xyz.shape[0]), num_samples=k, replacement=False
            )
        elif prune_type in ('gradient_all', 'gradient_oneshot'):
            score = scores if scores is not None else self._get_gradient_scores()
            indices = torch.topk(score, k=k, largest=largest)[1].squeeze()
        else:
            raise ValueError(f'Unknown prune_type: {prune_type}')

        if largest:
            mask = torch.zeros(self._xyz.shape[0], dtype=torch.bool)
            mask[indices] = True
        else:
            mask = torch.ones(self._xyz.shape[0], dtype=torch.bool)
            mask[indices] = False
        return mask, None

    # ------------------------------------------------------------------
    # Morton-order sort (for compression locality)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _sort_morton(self):
        xyz_range = self._xyz.max(0).values - self._xyz.min(0).values + 1e-7
        xyz_q = ((2**21 - 1) * (self._xyz - self._xyz.min(0).values) / xyz_range).long()
        order = mortonEncode(xyz_q).sort().indices
        self.index_primitives(order, requires_grad=False)
        return order

    # ------------------------------------------------------------------
    # Compression payloads (container format: gode/streams.py)
    # ------------------------------------------------------------------

    @staticmethod
    def _quantize(tensor, qa, dtype):
        return torch.quantize_per_tensor(tensor, qa.scale, qa.zero_point, dtype).int_repr().cpu().numpy()

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

        With quantize=False every attribute is stored as float16, which is what
        the no-QAT ablation reports.
        """
        geom = {
            "quantized":  quantize,
            "rotation_n": int(self._rotation.shape[0]),
            "xyz":        self._xyz.detach().half().cpu().numpy(),
            "scaling":    self._scaling.detach().half().cpu().numpy(),
        }
        if self._sb_params.numel() > 0:
            geom["sb_params"] = self._sb_params.detach().half().cpu().numpy()

        if quantize:
            small3, rot_idx = _quat_encode_smallest3(self._rotation.detach().cpu().numpy())
            geom.update(
                opacity=self._quantize(self.opacity_activation(self._opacity.detach()),
                                       self.opacity_qa, torch.quint8),
                opacity_scale=float(self.opacity_qa.scale),
                opacity_zp=int(self.opacity_qa.zero_point),
                beta=self._quantize(self._beta.detach(), self.beta_qa, torch.qint8),
                beta_scale=float(self.beta_qa.scale),
                beta_zp=int(self.beta_qa.zero_point),
                rotation=self._quantize(torch.from_numpy(small3).to(self._rotation.device),
                                        self.rotation_qa, torch.qint8),
                rotation_scale=float(self.rotation_qa.scale),
                rotation_zp=int(self.rotation_qa.zero_point),
                rotation_idx=_pack_rot_idx(rot_idx),
            )
        else:
            geom.update(
                opacity=self._opacity.detach().half().cpu().numpy(),
                beta=self._beta.detach().half().cpu().numpy(),
                rotation=self._rotation.detach().half().cpu().numpy(),
            )

        features = {"color_representation": self.color_representation, "quantized": quantize}
        if self.color_representation == "voronoi":
            sites = F.normalize(self._sites.detach(), dim=-1)
            features["num_lobes"] = int(self._sites.shape[1])
            if quantize:
                features.update(
                    sites=self._quantize(_oct_encode(sites), self.sites_qa, torch.qint8),
                    sites_scale=float(self.sites_qa.scale),
                    sites_zp=int(self.sites_qa.zero_point),
                    colors=self._quantize(self._colors.detach(), self.colors_qa, torch.qint8),
                    colors_scale=float(self.colors_qa.scale),
                    colors_zp=int(self.colors_qa.zero_point),
                    log_tau=self._quantize(self._log_tau_sites.detach(), self.log_tau_qa, torch.qint8),
                    log_tau_scale=float(self.log_tau_qa.scale),
                    log_tau_zp=int(self.log_tau_qa.zero_point),
                )
            else:
                features.update(
                    sites=sites.half().cpu().numpy(),
                    colors=self._colors.detach().half().cpu().numpy(),
                    log_tau=self._log_tau_sites.detach().half().cpu().numpy(),
                )
        elif self.color_representation in ("sh", "sb"):
            if quantize:
                features.update(
                    sh0=self._quantize(self._sh0.detach(), self.sh0_qa, self.sh0_qa.dtype),
                    sh0_scale=float(self.sh0_qa.scale),
                    sh0_zp=int(self.sh0_qa.zero_point),
                    shN=self._quantize(self._shN.detach(), self.shN_qa, self.shN_qa.dtype),
                    shN_scale=float(self.shN_qa.scale),
                    shN_zp=int(self.shN_qa.zero_point),
                )
            else:
                features.update(
                    sh0=self._sh0.detach().half().cpu().numpy(),
                    shN=self._shN.detach().half().cpu().numpy(),
                )
        else:
            raise NotImplementedError(
                f"Compression is not implemented for color_representation="
                f"'{self.color_representation}'"
            )
        return {"geom": geom, "features": features}

    @torch.no_grad()
    def decode_payload(self, payload):
        """Inverse of `encode_payload`; replaces the primitives in place."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        geom, features = payload["geom"], payload["features"]

        self.color_representation = features.get("color_representation", self.color_representation)
        self._xyz     = self._fp16(geom, "xyz", device)
        self._scaling = self._fp16(geom, "scaling", device)
        if "sb_params" in geom:
            self._sb_params = self._fp16(geom, "sb_params", device)

        if geom["quantized"]:
            opacity = self._dequantize(geom, "opacity", device).clamp(1e-6, 1 - 1e-6)
            self._opacity = inverse_sigmoid(opacity)
            self._beta = self._dequantize(geom, "beta", device)
            small3 = self._dequantize(geom, "rotation", device).cpu().numpy()
            rot_idx = _unpack_rot_idx(np.ascontiguousarray(geom["rotation_idx"]), geom["rotation_n"])
            rotation = _quat_decode_smallest3(small3.reshape(-1, 3), rot_idx)
            self._rotation = torch.from_numpy(rotation).float().to(device)
        else:
            self._opacity  = self._fp16(geom, "opacity", device)
            self._beta     = self._fp16(geom, "beta", device)
            self._rotation = self._fp16(geom, "rotation", device)

        if self.color_representation == "voronoi":
            self.num_lobes = int(features["num_lobes"])
            if features["quantized"]:
                sites = _oct_decode(self._dequantize(features, "sites", device))
                colors = self._dequantize(features, "colors", device)
                log_tau = self._dequantize(features, "log_tau", device)
            else:
                sites = self._fp16(features, "sites", device)
                colors = self._fp16(features, "colors", device)
                log_tau = self._fp16(features, "log_tau", device)
            self._sites         = nn.Parameter(sites, requires_grad=True)
            self._colors        = nn.Parameter(colors, requires_grad=True)
            self._log_tau_sites = nn.Parameter(log_tau, requires_grad=True)
        else:
            if features["quantized"]:
                sh0 = self._dequantize(features, "sh0", device)
                shN = self._dequantize(features, "shN", device)
            else:
                sh0 = self._fp16(features, "sh0", device)
                shN = self._fp16(features, "shN", device)
            self._sh0 = nn.Parameter(sh0, requires_grad=True)
            self._shN = nn.Parameter(shN, requires_grad=True)
        return True


    def add_new_gs(self, cap_max):
        current_num_points = self._opacity.shape[0]
        target_num = min(cap_max, int(1.05 * current_num_points))
        num_gs = max(0, target_num - current_num_points)
        if num_gs <= 0:
            return 0
        probs = self.get_opacity.squeeze(-1)
        add_idx, ratio = self._sample_alives(probs=probs, num=num_gs)

        (new_xyz, new_sh0, new_shN, new_sb_params, new_opacity, new_beta,
         new_scaling, new_rotation, *rest) = self._update_params(add_idx, ratio=ratio)

        params = {
            "xyz": new_xyz, "opacity": new_opacity, "scaling": new_scaling,
            "rotation": new_rotation, "sh0": new_sh0, "shN": new_shN,
            "sb_params": new_sb_params, "beta": new_beta,
        }
        if self.color_representation == "voronoi":
            params.update({"sites": rest[0], "colors": rest[1], "log_tau_sites": rest[2]})
        elif self.color_representation == "sgs":
            params.update({"mu": rest[0], "alpha": rest[1], "sharpness": rest[2]})

        self._opacity[add_idx] = new_opacity
        self.densification_postfix(params, reset_params=False)
        self.replace_tensors_to_optimizer(inds=add_idx)
        return num_gs

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, viewpoint_camera, render_mode="RGB", mask=None, measure_fps=False):

        def _m(t):
            return t if mask is None else t[mask]

        K = torch.zeros((3, 3), device=viewpoint_camera.projection_matrix.device)
        fx = 0.5 * viewpoint_camera.image_width  / math.tan(viewpoint_camera.FoVx / 2)
        fy = 0.5 * viewpoint_camera.image_height / math.tan(viewpoint_camera.FoVy / 2)
        K[0, 0] = fx; K[1, 1] = fy
        K[0, 2] = viewpoint_camera.image_width  / 2
        K[1, 2] = viewpoint_camera.image_height / 2
        K[2, 2] = 1.0

        xyz = _m(self.get_xyz)

        if self.color_representation in ("voronoi", "sgs"):
            dir_pp = xyz - viewpoint_camera.camera_center.repeat(xyz.shape[0], 1)
            # Some scene/camera pairs contain points at (or numerically extremely
            # close to) the camera center. Manual division turns their zero-length
            # view vectors into NaNs, which poison the spherical-Voronoi backward.
            dir_pp_normalized = torch.nan_to_num(
                F.normalize(dir_pp, dim=1, eps=1e-8),
                nan=0.0, posinf=0.0, neginf=0.0,
            )
            colors_precomp = self.compute_radiance(dir_pp_normalized, mask=mask)
            active_degree  = None
        else:  # sh or sb
            colors_precomp = _m(self.get_shs)
            active_degree  = self.active_sh_degree

        sb_params = _m(self.get_sb_params) if self.get_sb_params.numel() > 0 else self.get_sb_params

        if measure_fps:
            torch.cuda.synchronize()
            t0 = time.time()

        rgbs, alphas, meta = rasterization(
            means=xyz, quats=_m(self.get_rotation),
            scales=_m(self.get_scaling), opacities=_m(self.get_opacity).squeeze(),
            betas=_m(self.get_beta).squeeze(), colors=colors_precomp,
            viewmats=viewpoint_camera.world_view_transform.transpose(0, 1).unsqueeze(0),
            Ks=K.unsqueeze(0), width=viewpoint_camera.image_width,
            height=viewpoint_camera.image_height,
            backgrounds=self.background.unsqueeze(0),
            render_mode=render_mode, covars=None, sh_degree=active_degree,
            sb_number=self.sb_number, sb_params=sb_params, packed=False,
        )

        fps = None
        if measure_fps:
            torch.cuda.synchronize()
            elapsed = time.time() - t0
            fps = 1.0 / elapsed if elapsed > 0 else float('inf')

        rgb = rgbs.permute(0, 3, 1, 2).contiguous()[0]

        result = {
            "render":            rgb,
            "viewspace_points":  meta["means2d"],
            "visibility_filter": meta["radii"] > 0,
            "radii":             meta["radii"],
            "is_used":           meta["radii"] > 0,
        }
        if fps is not None:
            result["fps"] = fps
        return result
