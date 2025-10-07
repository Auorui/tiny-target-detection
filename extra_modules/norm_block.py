import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import autopad

class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU, inplace=True, bias=True, apply_act=True,):
        super(ConvNormAct, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.apply_act = apply_act
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size, stride,
                              padding=(kernel_size - 1) // 2 if padding is None else padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.norm = norm_layer(out_channels) if norm_layer else None
        self.act = nn.Identity() if activation_layer is None else activation_layer(inplace)

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


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x