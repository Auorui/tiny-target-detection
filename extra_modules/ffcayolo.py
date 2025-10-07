import torch
import torch.nn as nn
from ultralytics.nn.extra_modules.norm_block import ConvNormAct, ConvWithoutBN
from ultralytics.nn.modules.conv import Conv


class FEM(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, scale=0.1, map_reduce=8):
        super(FEM, self).__init__()
        self.scale = scale
        self.out_channels = out_channels
        inter_planes = in_channels // map_reduce
        self.branch0 = nn.Sequential(
            ConvNormAct(in_channels, 2 * inter_planes, kernel_size=1, stride=stride),
            ConvNormAct(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1, apply_act=False)
        )
        self.branch1 = nn.Sequential(
            ConvNormAct(in_channels, inter_planes, kernel_size=1, stride=1),
            ConvNormAct(inter_planes, (inter_planes // 2) * 3, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            ConvNormAct((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            ConvNormAct(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, apply_act=False)
        )
        self.branch2 = nn.Sequential(
            ConvNormAct(in_channels, inter_planes, kernel_size=1, stride=1),
            ConvNormAct(inter_planes, (inter_planes // 2) * 3, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            ConvNormAct((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            ConvNormAct(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, apply_act=False)
        )

        self.ConvLinear = ConvNormAct(6 * inter_planes, out_channels, kernel_size=1, stride=1, apply_act=False)
        self.shortcut = ConvNormAct(in_channels, out_channels, kernel_size=1, stride=stride, apply_act=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)
        return out

class FFMConcat(nn.Module):
    def __init__(self, total_channel, eps=1e-4):
        """
        total_channel: sum of all input channels
        eps: A small constant to prevent division by zero.
        """
        super(FFMConcat, self).__init__()
        self.eps = eps
        # The total sum of external input channels
        self.total_channel = total_channel
        # Learnable weights (one weight per channel)
        self.w = nn.Parameter(torch.ones(total_channel, dtype=torch.float32), requires_grad=True)

    def forward(self, x: list):
        """
        x: list of feature maps [x1, x2, ..., xN]
        """
        B, _, H, W = x[0].shape
        # Check if the total number of channels matches.
        channels = [xi.shape[1] for xi in x]
        assert sum(channels) == self.total_channel, \
            f"The sum of input channels {sum(channels)} does not match the defined total_channel={self.total_channel}"
        weight = self.w / (torch.sum(self.w, dim=0) + self.eps)
        outs = []
        start = 0
        for i, xi in enumerate(x):
            # Check if the space dimensions are consistent.
            assert xi.shape[2:] == (H, W), f"The size of the {i}th input space {xi.shape[2:]} does not match the first input {H, W}."
            ci = xi.shape[1]
            wi = weight[start:start + ci]
            # First move the channel to the end, then multiply them channel by channel, and finally move it back.
            xi_weighted = (wi * xi.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            outs.append(xi_weighted)
            start += ci
        return torch.cat(outs, dim=1)

class SCAM(nn.Module):
    def __init__(self, in_channels):
        super(SCAM, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels

        self.k = Conv(in_channels, 1, 1, 1)
        self.v = Conv(in_channels, self.inter_channels, 1, 1)
        self.m = ConvWithoutBN(self.inter_channels, in_channels, 1, 1)
        self.m2 = Conv(2, 1, 1, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # GAP
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # GMP

    def forward(self, x):
        n, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3)
        avg = self.avg_pool(x).softmax(1).view(n, 1, 1, c)
        max = self.max_pool(x).softmax(1).view(n, 1, 1, c)
        k = self.k(x).view(n, 1, -1, 1).softmax(2)       # [N, 1, HW, 1]
        v = self.v(x).view(n, 1, c, -1)                  # [N, 1, C, HW]
        y = torch.matmul(v, k).view(n, c, 1, 1)          # [N, C, 1, 1]
        y_avg = torch.matmul(avg, v).view(n, 1, h, w)
        y_max = torch.matmul(max, v).view(n, 1, h, w)
        y_cat = torch.cat((y_avg, y_max), 1)              # [N, 2, H, W]
        y = self.m(y) * self.m2(y_cat).sigmoid()
        return x + y

if __name__=="__main__":
    in_channels = 64
    out_channels = 128
    batch_size = 4
    height, width = 32, 32

    fem = FEM(in_channels=in_channels, out_channels=out_channels)
    x = torch.randn(batch_size, in_channels, height, width)
    print(f"FEM input shape: {x.shape}")

    output = fem(x)
    print(f"FEM output shape: {output.shape}")

    c1, c2 = 32, 32
    ffm2 = FFMConcat2(dimension=1, channel_1=c1, channel_2=c2)
    x1 = torch.randn(batch_size, c1, height, width)
    x2 = torch.randn(batch_size, c2, height, width)
    out_ffm2 = ffm2([x1, x2])
    print(f"FFMConcat2 input shape: {[x1.shape, x2.shape]}")
    print(f"FFMConcat2 output shape: {out_ffm2.shape}\n")

    c3 = 32
    ffm3 = FFMConcat3(dimension=1, channel_1=c1, channel_2=c2, channel_3=c3)
    x3 = torch.randn(batch_size, c3, height, width)
    out_ffm3 = ffm3([x1, x2, x3])
    print(f"FFMConcat3 input shape: {[x1.shape, x2.shape, x3.shape]}")
    print(f"FFMConcat3 output shape: {out_ffm3.shape}\n")
    print(out_ffm3[0][0])


    total_channels = c1 + c2 + c3
    ffm = FFMConcat(total_channel=total_channels)
    out_ffm = ffm([x1, x2, x3])
    print(out_ffm[0][0])
    print(f"FFMConcat input shape: {[x1.shape, x2.shape, x3.shape]}")
    print(f"FFMConcat output shape: {out_ffm.shape}\n")


    scam_in_channels = 128
    scam = SCAM(in_channels=scam_in_channels)
    x_scam = torch.randn(batch_size, scam_in_channels, height, width)
    out_scam = scam(x_scam)
    print(f"SCAM input shape: {x_scam.shape}")
    print(f"SCAM output shape: {out_scam.shape}\n")