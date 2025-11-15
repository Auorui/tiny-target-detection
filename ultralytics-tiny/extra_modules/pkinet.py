import torch
import torch.nn as nn
from timm.layers import DropPath
from ultralytics.nn.extra_modules import ConvWithoutBN, BCHW2BHWC, BHWC2BCHW, make_divisible
from ultralytics.nn.modules.conv import autopad, Conv

class GSiLU(nn.Module):
    """Global Sigmoid-Gated Linear Unit, reproduced from paper <SIMPLE CNN FOR VISION>"""
    def __init__(self):
        super().__init__()
        self.adpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return x * torch.sigmoid(self.adpool(x))

class CAA(nn.Module):
    """Context Anchor Attention"""
    def __init__(
            self,
            channels: int,
            h_kernel_size: int = 11,
            v_kernel_size: int = 11,
    ):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.h_conv = nn.Conv2d(channels, channels, (1, h_kernel_size), 1,
                                 (0, h_kernel_size // 2), groups=channels)
        self.v_conv = nn.Conv2d(channels, channels, (v_kernel_size, 1), 1,
                                 (v_kernel_size // 2, 0), groups=channels)
        self.conv2 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.act = nn.Sigmoid()

    def forward(self, x):
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        return attn_factor

class ConvFFN(nn.Module):
    """Multi-layer perceptron implemented with ConvModule"""
    def __init__(
            self,
            in_channels: int,
            out_channels = None,
            hidden_channels_scale: float = 4.0,
            hidden_kernel_size: int = 3,
            dropout_rate: float = 0.,
            add_identity: bool = True,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = int(in_channels * hidden_channels_scale)

        self.ffn_layers = nn.Sequential(
            BCHW2BHWC(),
            nn.LayerNorm(in_channels),
            BHWC2BCHW(),
            Conv(in_channels, hidden_channels, k=1, s=1, p=0),
            Conv(hidden_channels, hidden_channels, k=hidden_kernel_size, s=1,
                       p=hidden_kernel_size // 2, g=hidden_channels),
            GSiLU(),
            nn.Dropout(dropout_rate),
            Conv(hidden_channels, out_channels, k=1, s=1, p=0),
            nn.Dropout(dropout_rate),
        )
        self.add_identity = add_identity

    def forward(self, x):
        x = x + self.ffn_layers(x) if self.add_identity else self.ffn_layers(x)
        return x

class PKIStem(nn.Module):
    """Stem layer"""
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            expansion: float = 1.0,
    ):
        super().__init__()
        hidden_channels = make_divisible(int(out_channels * expansion), 8)
        self.down_conv = Conv(in_channels, hidden_channels, k=3, s=2, p=1)
        self.conv1 = Conv(hidden_channels, hidden_channels, k=3, s=1, p=1)
        self.conv2 = Conv(hidden_channels, out_channels, k=3, s=1, p=1)

    def forward(self, x):
        return self.conv2(self.conv1(self.down_conv(x)))


class DownSamplingLayer(nn.Module):
    """Down sampling layer"""
    def __init__(
            self,
            in_channels: int,
            out_channels = None,
    ):
        super().__init__()
        out_channels = out_channels or (in_channels * 2)
        self.down_conv = Conv(in_channels, out_channels, 3, 2, 1)

    def forward(self, x):
        return self.down_conv(x)


class InceptionBottleneck(nn.Module):
    """Bottleneck with Inception module"""
    def __init__(
            self,
            in_channels: int,
            out_channels = None,
            kernel_sizes = (3, 5, 7, 9, 11),
            dilations = (1, 1, 1, 1, 1),
            expansion: float = 1.0,
            add_identity: bool = True,
            with_caa: bool = True,
            caa_kernel_size: int = 11,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = make_divisible(int(out_channels * expansion), 8)
        self.pre_conv = Conv(in_channels, hidden_channels, 1, 1, 0, 1)
        self.dw_conv = ConvWithoutBN(hidden_channels, hidden_channels, kernel_sizes[0], 1,
                                  autopad(kernel_sizes[0], None, dilations[0]), dilations[0],
                                  groups=hidden_channels)
        self.dw_conv1 = ConvWithoutBN(hidden_channels, hidden_channels, kernel_sizes[1], 1,
                                   autopad(kernel_sizes[1], None, dilations[1]), dilations[1],
                                   groups=hidden_channels)
        self.dw_conv2 = ConvWithoutBN(hidden_channels, hidden_channels, kernel_sizes[2], 1,
                                   autopad(kernel_sizes[2], None, dilations[2]), dilations[2],
                                   groups=hidden_channels)
        self.dw_conv3 = ConvWithoutBN(hidden_channels, hidden_channels, kernel_sizes[3], 1,
                                   autopad(kernel_sizes[3], None, dilations[3]), dilations[3],
                                   groups=hidden_channels)
        self.dw_conv4 = ConvWithoutBN(hidden_channels, hidden_channels, kernel_sizes[4], 1,
                                   autopad(kernel_sizes[4], None, dilations[4]), dilations[4],
                                   groups=hidden_channels)
        self.pw_conv = Conv(hidden_channels, hidden_channels, 1, 1, 0, 1)
        if with_caa:
            self.caa_factor = CAA(hidden_channels, caa_kernel_size, caa_kernel_size)
        else:
            self.caa_factor = None
        self.add_identity = add_identity and in_channels == out_channels
        self.post_conv = Conv(hidden_channels, out_channels, 1, 1, 0, 1)

    def forward(self, x):
        x = self.pre_conv(x)
        y = x  # if there is an inplace operation of x, use y = x.clone() instead of y = x
        x = self.dw_conv(x)
        x = x + self.dw_conv1(x) + self.dw_conv2(x) + self.dw_conv3(x) + self.dw_conv4(x)
        x = self.pw_conv(x)
        if self.caa_factor is not None:
            y = self.caa_factor(y)
        if self.add_identity:
            y = x * y
            x = x + y
        else:
            x = x * y
        x = self.post_conv(x)
        return x

class PKIBlock(nn.Module):
    """Poly Kernel Inception Block"""
    def __init__(
            self,
            in_channels: int,
            out_channels = None,
            kernel_sizes = (3, 5, 7, 9, 11),
            dilations = (1, 1, 1, 1, 1),
            with_caa: bool = True,
            caa_kernel_size: int = 11,
            expansion: float = 1.0,
            ffn_scale: float = 4.0,
            ffn_kernel_size: int = 3,
            dropout_rate: float = 0.,
            drop_path_rate: float = 0.,
            layer_scale = 1.0,
            add_identity: bool = True,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = make_divisible(int(out_channels * expansion), 8)
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.norm2 = nn.BatchNorm2d(hidden_channels)
        self.block = InceptionBottleneck(in_channels, hidden_channels, kernel_sizes, dilations,
                                         expansion=1.0, add_identity=True,
                                         with_caa=with_caa, caa_kernel_size=caa_kernel_size)
        self.ffn = ConvFFN(hidden_channels, out_channels, ffn_scale, ffn_kernel_size, dropout_rate, add_identity=False)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        self.layer_scale = layer_scale
        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(hidden_channels), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(out_channels), requires_grad=True)
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x):
        if self.layer_scale:
            if self.add_identity:
                x = x + self.drop_path(self.gamma1.unsqueeze(-1).unsqueeze(-1) * self.block(self.norm1(x)))
                x = x + self.drop_path(self.gamma2.unsqueeze(-1).unsqueeze(-1) * self.ffn(self.norm2(x)))
            else:
                x = self.drop_path(self.gamma1.unsqueeze(-1).unsqueeze(-1) * self.block(self.norm1(x)))
                x = self.drop_path(self.gamma2.unsqueeze(-1).unsqueeze(-1) * self.ffn(self.norm2(x)))
        else:
            if self.add_identity:
                x = x + self.drop_path(self.block(self.norm1(x)))
                x = x + self.drop_path(self.ffn(self.norm2(x)))
            else:
                x = self.drop_path(self.block(self.norm1(x)))
                x = self.drop_path(self.ffn(self.norm2(x)))
        return x

class PKIStage(nn.Module):
    """Poly Kernel Inception Stage"""
    def __init__(
            self,
            in_channels,
            out_channels,
            num_blocks,
            shortcut_ffn_scale=4.0,
            shortcut_ffn_kernel_size=5,
            kernel_sizes = (3, 5, 7, 9, 11),
            dilations = (1, 1, 1, 1, 1),
            expansion = 0.5,
            ffn_scale = 4.0,
            ffn_kernel_size = 3,
            dropout_rate = 0.1,
            drop_path_rate = 0.1,
            layer_scale = 1.0,
            shortcut_with_ffn = True,
            add_identity = True,
            with_caa = True,
            caa_kernel_size = 11,
    ):
        super().__init__()
        hidden_channels = make_divisible(int(out_channels * expansion), 8)

        self.downsample = DownSamplingLayer(in_channels, out_channels)

        self.conv1 = Conv(out_channels, 2 * hidden_channels, 1, 1, 0, d=1)
        self.conv2 = Conv(2 * hidden_channels, out_channels, 1, 1, 0, d=1)
        self.conv3 = Conv(out_channels, out_channels, 1, 1, 0, d=1)

        self.ffn = ConvFFN(hidden_channels, hidden_channels, shortcut_ffn_scale, shortcut_ffn_kernel_size, 0.,
                           add_identity=True) if shortcut_with_ffn else None

        self.blocks = nn.ModuleList([
            PKIBlock(hidden_channels, hidden_channels, kernel_sizes, dilations, with_caa,
                     caa_kernel_size+2*i, 1.0, ffn_scale, ffn_kernel_size, dropout_rate,
                     drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                     layer_scale, add_identity) for i in range(num_blocks)
        ])

    def forward(self, x):
        x = self.downsample(x)

        x, y = list(self.conv1(x).chunk(2, 1))
        if self.ffn is not None:
            x = self.ffn(x)

        z = [x]
        t = torch.zeros(y.shape, device=y.device)
        for block in self.blocks:
            t = t + block(y)
        z.append(t)
        z = torch.cat(z, dim=1)
        z = self.conv2(z)
        z = self.conv3(z)
        return z

class PKINet(nn.Module):
    def __init__(
            self,
            in_channels=16,
            num_blocks=(4, 14, 22, 4),
            kernel_sizes=(3, 5, 7, 9, 11),
            drop_path_rate=0.1,
            shortcut_ffn_kernel_size=(5, 7, 9, 11),
    ):
        super(PKINet, self).__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_blocks))]
        self.stem = PKIStem(3, in_channels, expansion=1.0)
        self.stage_1 = PKIStage(
            in_channels, in_channels * 2, num_blocks[0],
            kernel_sizes=kernel_sizes,
            drop_path_rate=dpr[0:num_blocks[0]],
            shortcut_ffn_scale=8.0,
            shortcut_ffn_kernel_size=shortcut_ffn_kernel_size[0]
        )
        self.stage_2 = PKIStage(
            in_channels * 2, in_channels * 4, num_blocks[1],
            kernel_sizes=kernel_sizes,
            drop_path_rate=dpr[num_blocks[0]:num_blocks[0] + num_blocks[1]],
            shortcut_ffn_scale=8.0,
            shortcut_ffn_kernel_size=shortcut_ffn_kernel_size[1]
        )
        self.stage_3 = PKIStage(
            in_channels * 4, in_channels * 8, num_blocks[2],
            kernel_sizes=kernel_sizes,
            drop_path_rate=dpr[num_blocks[0] + num_blocks[1]:num_blocks[0] + num_blocks[1] + num_blocks[2]],
            shortcut_ffn_scale=4.0,
            shortcut_ffn_kernel_size=shortcut_ffn_kernel_size[2]
        )
        self.stage_4 = PKIStage(
            in_channels * 8, in_channels * 16, num_blocks[3],
            kernel_sizes=kernel_sizes,
            drop_path_rate=dpr[sum(num_blocks[:3]):sum(num_blocks)],
            shortcut_ffn_scale=4.0,
            shortcut_ffn_kernel_size=shortcut_ffn_kernel_size[3]
        )

    def forward(self, x):
        x = self.stem(x)
        x1 = self.stage_1(x)
        x2 = self.stage_2(x1)
        x3 = self.stage_3(x2)
        x4 = self.stage_4(x3)
        return x4


def PKINet_t(in_channels=16, num_blocks=(4, 14, 22, 4)):
    return PKINet(in_channels, num_blocks)

def PKINet_s(in_channels=32, num_blocks=(4, 12, 20, 4)):
    return PKINet(in_channels, num_blocks)

def PKINet_b(in_channels=40, num_blocks=(6, 16, 24, 6)):
    return PKINet(in_channels, num_blocks)


if __name__=="__main__":
    model = PKINet_b()

    test_sizes = [(224, 224), (320, 320), (512, 512)]
    for h, w in test_sizes:
        print(f"\nTesting input size: {h}x{w}")
        x = torch.randn(1, 3, h, w)
        outputs = model(x)
        print(f"Output shapes:{outputs.shape}")
