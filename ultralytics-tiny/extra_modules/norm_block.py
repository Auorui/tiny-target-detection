import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import autopad
from itertools import repeat
import collections.abc


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU, bias=True, apply_act=True,):
        super(ConvNormAct, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.apply_act = apply_act
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size, stride,
                              padding=(kernel_size - 1) // 2 if padding is None else padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.norm = norm_layer(out_channels) if norm_layer else None
        self.act = nn.Identity() if activation_layer is None else activation_layer()

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


class BCHW2BHWC(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        return x.permute([0, 2, 3, 1])


class BHWC2BCHW(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        return x.permute([0, 3, 1, 2])

def make_divisible(value, divisor, min_value=None, min_ratio=0.9):
    """Make divisible function.

    This function rounds the channel number to the nearest value that can be
    divisible by the divisor. It is taken from the original tf repo. It ensures
    that all layers have a channel number that is divisible by divisor. It can
    be seen here: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py  # noqa

    Args:
        value (int): The original channel number.
        divisor (int): The divisor to fully divide the channel number.
        min_value (int): The minimum value of the output channel.
            Default: None, means that the minimum value equal to the divisor.
        min_ratio (float): The minimum ratio of the rounded channel number to
            the original channel number. Default: 0.9.

    Returns:
        int: The modified output channel number.
    """

    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than (1-min_ratio).
    if new_value < min_ratio * value:
        new_value += divisor
    return new_value


class PartialConv2d(nn.Module):
    def __init__(self, dim, n_div, kernel_size=3, forward='split_cat'):
        """
        PartialConv
        code from: https://github.com/JierunChen/FasterNet/blob/master/models/fasternet.py
        """
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, kernel_size, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x