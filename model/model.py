import functools

import torch
import torch.nn as nn

from .layer_utils import get_norm_layer, ResNetBlock
from base.base_model import BaseModel


class ResNet(BaseModel):
    """Define a generator using ResNet"""

    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9, norm_type='instance', padding_type='reflect',
                 use_dropout=True, learn_residual=True):
        super(ResNet, self).__init__()

        self.learn_residual = learn_residual

        norm_layer = get_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        sequence = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        ]

        n_downsampling = 2
        for i in range(n_downsampling):  # downsample the feature map
            mult = 2 ** i
            sequence += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)
            ]

        for i in range(n_blocks):  # ResNet
            sequence += [
                ResNetBlock(ngf * 2 ** n_downsampling, norm_layer, padding_type, use_dropout, use_bias)
            ]

        for i in range(n_downsampling):  # upsample the feature map
            mult = 2 ** (n_downsampling - i)
            sequence += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                   output_padding=1, bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]

        sequence += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = self.model(x)
        if self.learn_residual:
            out = x + out
            out = torch.clamp(out, min=-1, max=1)  # clamp to [-1,1] according to normalization(mean=0.5, var=0.5)
        return out

