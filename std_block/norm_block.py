import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import autopad

class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU, inplace=True, bias=True, apply_act=True,):
        super(ConvNormAct, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.apply_act = apply_act
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size, stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.norm = norm_layer(out_channels) if norm_layer else None
        self.act = activation_layer(inplace) if activation_layer else None

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act and self.apply_act:
            x = self.act(x)
        return x

class ConvWithoutBN(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    def __init__(self, c1, c2, kernel_size=1, stride=1, padding=None, dilation=1, groups=1, activation_layer=nn.SiLU(), apply_act=True, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size, stride, autopad(kernel_size, padding, dilation), groups=groups, dilation=dilation, bias=bias)
        self.act = activation_layer if apply_act is True else apply_act if isinstance(apply_act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x))