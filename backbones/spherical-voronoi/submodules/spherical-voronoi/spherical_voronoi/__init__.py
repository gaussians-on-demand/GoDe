import torch
from . import _C


class _SV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dirs, sites, tau, colors):
        rgb, w, dist, feat = _C.forward(dirs, sites, tau, colors)
        ctx.save_for_backward(dirs, sites, tau, colors, w, dist, feat)
        return rgb

    @staticmethod
    def backward(ctx, v_rgb):
        dirs, sites, tau, colors, w, dist, feat = ctx.saved_tensors
        v_sites, v_tau, v_colors = _C.backward(
            dirs, sites, tau, colors, w, dist, feat, v_rgb.contiguous()
        )
        return None, v_sites, v_tau, v_colors


def spherical_voronoi(
    dirs: torch.Tensor,    # [N, 3] unit view directions
    sites: torch.Tensor,   # [N, S, 3] unit Voronoi sites
    tau: torch.Tensor,     # [N, S] temperatures (already exp'd)
    colors: torch.Tensor,  # [N, S, 3]
) -> torch.Tensor:         # [N, 3]  rgb = relu(weighted_sum + 0.5)
    return _SV.apply(
        dirs.contiguous(),
        sites.contiguous(),
        tau.contiguous(),
        colors.contiguous(),
    )
