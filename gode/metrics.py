"""Image metrics used to report every level, kept inside the core so that it
stays independent of the backbone repository."""

from math import exp

import torch
import torch.nn.functional as F

from gode.perceptual import LPIPS

_LPIPS_CACHE = {}


def psnr(image, gt):
    mse = ((image - gt) ** 2).view(image.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def _gaussian_window(window_size, channels, sigma=1.5):
    coords = torch.tensor([exp(-((x - window_size // 2) ** 2) / (2 * sigma ** 2))
                           for x in range(window_size)])
    coords = (coords / coords.sum()).unsqueeze(1)
    window = coords.mm(coords.t()).float().unsqueeze(0).unsqueeze(0)
    return window.expand(channels, 1, window_size, window_size).contiguous()


def ssim(image, gt, window_size=11):
    channels = image.size(-3)
    window = _gaussian_window(window_size, channels).to(image.device).type_as(image)
    padding, groups = window_size // 2, channels

    mu1 = F.conv2d(image, window, padding=padding, groups=groups)
    mu2 = F.conv2d(gt, window, padding=padding, groups=groups)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2

    sigma1_sq = F.conv2d(image * image, window, padding=padding, groups=groups) - mu1_sq
    sigma2_sq = F.conv2d(gt * gt, window, padding=padding, groups=groups) - mu2_sq
    sigma12 = F.conv2d(image * gt, window, padding=padding, groups=groups) - mu1_mu2

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = (((2 * mu1_mu2 + c1) * (2 * sigma12 + c2))
                / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)))
    return ssim_map.mean()


def lpips_model(device, net_type="vgg"):
    """Cached LPIPS criterion; inputs are expected in [-1, 1]."""
    key = (net_type, str(device))
    if key not in _LPIPS_CACHE:
        _LPIPS_CACHE[key] = LPIPS(net_type=net_type).to(device).eval()
    return _LPIPS_CACHE[key]
