import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv, C2f

class PConv(nn.Module):
    ''' Pinwheel-shaped Convolution using the Asymmetric Padding method. '''
    def __init__(self, c1, c2, k, s, p=None):
        super().__init__()
        # self.k = k
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)
        self.cat = Conv(c2, c2, 2, s=1, p=0)

    def forward(self, x):
        yw0 = self.cw(self.pad[0](x))
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))
        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))

class APBottleneck(nn.Module):
    """Asymmetric Padding bottleneck."""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        p = [(2, 0, 2, 0), (0, 2, 0, 2), (0, 2, 2, 0), (2, 0, 0, 2)]
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
        self.cv1 = Conv(c1, c_ // 4, k[0], 1, p=0)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2((torch.cat([self.cv1(self.pad[g](x)) for g in range(4)], 1))) if self.add else self.cv2(
            (torch.cat([self.cv1(self.pad[g](x)) for g in range(4)], 1)))

class APC2f(C2f):
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super(APC2f, self).__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            APBottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
