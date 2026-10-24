"""LPIPS, vendored so that the core carries no backbone dependency.

Adapted from https://github.com/S-aiueo32/lpips-pytorch, which reproduces
Zhang et al., "The Unreasonable Effectiveness of Deep Features as a
Perceptual Metric" (CVPR 2018) with the original pretrained weights.
"""

from collections import OrderedDict
from itertools import chain

import torch
import torch.nn as nn
from torchvision import models

WEIGHTS_URL = ("https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/"
               "master/lpips/weights/v{version}/{net_type}.pth")


def _normalize(x, eps=1e-10):
    return x / (torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True)) + eps)


def _linear_weights(net_type, version):
    state = torch.hub.load_state_dict_from_url(
        WEIGHTS_URL.format(version=version, net_type=net_type), progress=True,
        map_location=None if torch.cuda.is_available() else torch.device("cpu"))
    return OrderedDict((k.replace("lin", "").replace("model.", ""), v)
                       for k, v in state.items())


class _BaseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None])
        self.register_buffer("std", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None])

    def freeze(self):
        for param in chain(self.parameters(), self.buffers()):
            param.requires_grad = False

    def forward(self, x):
        x = (x - self.mean) / self.std
        output = []
        for i, layer in enumerate(self.layers, 1):
            x = layer(x)
            if i in self.target_layers:
                output.append(_normalize(x))
            if len(output) == len(self.target_layers):
                break
        return output


class _VGG16(_BaseNet):
    def __init__(self):
        super().__init__()
        self.layers = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.target_layers = [4, 9, 16, 23, 30]
        self.n_channels_list = [64, 128, 256, 512, 512]
        self.freeze()


class _AlexNet(_BaseNet):
    def __init__(self):
        super().__init__()
        self.layers = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).features
        self.target_layers = [2, 5, 8, 10, 12]
        self.n_channels_list = [64, 192, 384, 256, 256]
        self.freeze()


class LPIPS(nn.Module):
    """Perceptual distance between two images in [-1, 1]."""

    def __init__(self, net_type="vgg", version="0.1"):
        super().__init__()
        networks = {"vgg": _VGG16, "alex": _AlexNet}
        if net_type not in networks:
            raise NotImplementedError(f"net_type must be one of {sorted(networks)}")
        self.net = networks[net_type]()
        self.lin = nn.ModuleList(
            nn.Sequential(nn.Identity(), nn.Conv2d(channels, 1, 1, 1, 0, bias=False))
            for channels in self.net.n_channels_list)
        self.lin.load_state_dict(_linear_weights(net_type, version))
        for param in self.lin.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        features_x, features_y = self.net(x), self.net(y)
        differences = [(fx - fy) ** 2 for fx, fy in zip(features_x, features_y)]
        scores = [layer(d).mean((2, 3), True) for d, layer in zip(differences, self.lin)]
        return torch.sum(torch.cat(scores, 0), 0, True)
