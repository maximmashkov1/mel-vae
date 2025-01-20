import torch
from torch import nn
import numpy as np
import torch
from torch import nn
import math
import torch.nn.functional as F
import numpy as np


class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed


class Downsample(nn.Module):
    def __init__(self, down_type='full'):
        super().__init__()
        self.down_type=down_type

    def forward(self, x):
        if self.down_type=='full':
            return torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        elif self.down_type=='freq':
            return torch.nn.functional.avg_pool2d(x, kernel_size=(2,1), stride=(2,1))

class Upsample(nn.Module):
    def __init__(self, up_type='full'):
        super().__init__()
        self.up_type = up_type

    def forward(self, x, target_shape=None):
        if target_shape==None:
            target_shape = list(x.shape[-2:]) 
            target_shape[0] *= 2
            target_shape[1] *= (2 if self.up_type=='full' else 1)
        return torch.nn.functional.interpolate(x, size=target_shape, mode="bilinear")


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, is_discr=False, dropout=0, group_size=16):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        num_groups=max(1,int(in_channels/group_size))
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)
        
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.af = nn.LeakyReLU(0.02 if not is_discr else 0.1)

        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()


    def forward(self, x):
        h=x
        h = self.norm1(h)
        h = self.conv1(h)
        h = self.af(h)

        h = self.norm2(h)
        h = self.conv2(h)
        h = self.af(h)
        h = self.dropout(h)

        return self.shortcut(x) + h

