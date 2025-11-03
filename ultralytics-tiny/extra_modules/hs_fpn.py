# HS-FPN: High Frequency and Spatial Perception FPN for Tiny Object Detection
import torch
import numpy as np
import torch.nn as nn
import torch_dct as DCT
import torch.nn.functional as F
from einops import rearrange
from ultralytics.nn.extra_modules.norm_block import ConvWithoutBN, ConvNormAct, to_2tuple

class DctSpatialInteraction(nn.Module):
    def __init__(self,
                in_channels,
                ratio,
                isdct):
        super().__init__()
        """Spatial Path of HFP.Only p1&p2 use dct to extract high_frequency response"""
        self.ratio = ratio
        self.isdct = isdct   # true when in p1&p2 # false when in p3&p4
        if not self.isdct:
            self.spatial1x1 = nn.Sequential(
                ConvWithoutBN(in_channels, 1, kernel_size=1, bias=False, activation_layer=nn.ReLU(inplace=True)),
            )

    def forward(self, x):
        _, _, h0, w0 = x.size()
        if not self.isdct:
            return x * torch.sigmoid(self.spatial1x1(x))
        idct = DCT.dct_2d(x, norm='ortho')
        weight = self._compute_weight(h0, w0, self.ratio).to(x.device)
        weight = weight.view(1, h0, w0).expand_as(idct)
        dct = idct * weight  # filter out low-frequency features
        dct_ = DCT.idct_2d(dct, norm='ortho')  # generate spatial mask
        return x * dct_

    def _compute_weight(self, h, w, ratio):
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0
        return weight

class DctChannelInteraction(nn.Module):
    def __init__(self,
                 in_channels,
                 patch,
                 ratio,
                 isdct,
                 ):
        super(DctChannelInteraction, self).__init__()
        """Channel Path of HFP.Only p1&p2 use dct to extract high_frequency response"""
        self.in_channels = in_channels
        # 处理patch为None的情况
        if patch is None:
            self.h = self.w = 1  # 默认值
        else:
            self.h = patch[0]
            self.w = patch[1]
        self.ratio = ratio
        self.isdct = isdct
        self.channel1x1 = ConvWithoutBN(in_channels, in_channels, kernel_size=1, groups=32, bias=False, activation_layer=nn.ReLU(inplace=True))
        self.channel2x1 = ConvWithoutBN(in_channels, in_channels, kernel_size=1, groups=32, bias=False, activation_layer=nn.ReLU(inplace=True))
        self.relu = nn.ReLU()

    def forward(self, x):
        n, c, h, w = x.size()
        if not self.isdct:  # true when in p1&p2 # false when in p3&p4
            amaxp = F.adaptive_max_pool2d(x, output_size=(1, 1))
            aavgp = F.adaptive_avg_pool2d(x, output_size=(1, 1))
            channel = self.channel1x1(self.relu(amaxp)) + self.channel1x1(self.relu(aavgp))  # 2025 03 15 szc
            return x * torch.sigmoid(self.channel2x1(channel))

        idct = DCT.dct_2d(x, norm='ortho')
        weight = self._compute_weight(h, w, self.ratio).to(x.device)
        weight = weight.view(1, h, w).expand_as(idct)
        dct = idct * weight  # filter out low-frequency features
        dct_ = DCT.idct_2d(dct, norm='ortho')

        amaxp = F.adaptive_max_pool2d(dct_, output_size=(self.h, self.w))
        aavgp = F.adaptive_avg_pool2d(dct_, output_size=(self.h, self.w))
        amaxp = torch.sum(self.relu(amaxp), dim=[2, 3]).view(n, c, 1, 1)
        aavgp = torch.sum(self.relu(aavgp), dim=[2, 3]).view(n, c, 1, 1)

        # channel = torch.cat([self.channel1x1(aavgp), self.channel1x1(amaxp)], dim = 1)
        # TODO: The values of aavgp and amaxp appear to be on different scales. Add is a better choice instead of concate.
        channel = self.channel1x1(amaxp) + self.channel1x1(aavgp)  # 2025 03 15 szc
        return x * torch.sigmoid(self.channel2x1(channel))

    def _compute_weight(self, h, w, ratio):
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0
        return weight


class HFP(nn.Module):
    def __init__(self, c1, ratio=0, patch=1, isdct=0):
        super(HFP, self).__init__()
        """High Frequency Perception Module HFP"""
        ratio = to_2tuple(ratio)
        patch = to_2tuple(patch)
        self.spatial = DctSpatialInteraction(c1, ratio=ratio, isdct=isdct)
        self.channel = DctChannelInteraction(c1, patch=patch, ratio=ratio, isdct=isdct)
        self.out = nn.Sequential(
            ConvWithoutBN(c1, c1, kernel_size=3, padding=1,
                          activation_layer=nn.ReLU(inplace=True)),
            nn.GroupNorm(32, c1)
        )

    def forward(self, x):
        spatial = self.spatial(x)  # output of spatial path
        channel = self.channel(x)  # output of channel path
        return self.out(spatial + channel)


class SDP(nn.Module):
    def __init__(self, c1, inter_dim=None):
        """Spatial Dependency Perception Module SDP for YOLOv8
        Args:
            c1: input channels (自动从parse_model传递)
            inter_dim: intermediate dimension
        """
        super(SDP, self).__init__()
        self.inter_dim = inter_dim if inter_dim is not None else c1

        self.conv_q = nn.Sequential(
            ConvWithoutBN(c1, self.inter_dim, 1, padding=0, bias=False,
                          activation_layer=nn.ReLU(inplace=True)),
            nn.GroupNorm(32, self.inter_dim))
        self.conv_k = nn.Sequential(
            ConvWithoutBN(c1, self.inter_dim, 1, padding=0, bias=False,
                          activation_layer=nn.ReLU(inplace=True)),
            nn.GroupNorm(32, self.inter_dim))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # 解包输入
        if isinstance(x, (list, tuple)) and len(x) == 2:
            x_low, x_high = x
        else:
            return x

        b, c, h_low, w_low = x_low.size()
        _, _, h_high, w_high = x_high.size()

        # 自动计算patch_size - 基于两个特征图的比例
        # patch_size表示网格划分的数量，而不是每个patch的像素大小
        patch_h = h_high  # 网格行数 = 高层特征的高度
        patch_w = w_high  # 网格列数 = 高层特征的宽度

        # 计算每个patch的像素大小
        patch_size_h = h_low // patch_h  # 每个patch的高度
        patch_size_w = w_low // patch_w  # 每个patch的宽度

        # 确保能整除
        if h_low % patch_h != 0 or w_low % patch_w != 0:
            # 如果不能整除，调整特征图尺寸
            h_low_aligned = (h_low // patch_h) * patch_h
            w_low_aligned = (w_low // patch_w) * patch_w
            x_low = F.interpolate(x_low, size=(h_low_aligned, w_low_aligned), mode='nearest')
            x_high = F.interpolate(x_high, size=(h_low_aligned, w_low_aligned), mode='nearest')
            h_low, w_low = h_low_aligned, w_low_aligned

        # 重新计算patch的像素大小
        patch_size_h = h_low // patch_h
        patch_size_w = w_low // patch_w

        # 确保patch大小至少为1
        patch_size_h = max(1, patch_size_h)
        patch_size_w = max(1, patch_size_w)

        try:
            # 处理query (低层特征)
            # 将x_low划分为 patch_h x patch_w 个网格，每个网格大小为 patch_size_h x patch_size_w
            q = rearrange(self.conv_q(x_low),
                          'b c (h p1) (w p2) -> (b h w) (p1 p2) c',
                          p1=patch_size_h, p2=patch_size_w,
                          h=patch_h, w=patch_w)

            # 处理key (高层特征)
            # x_high已经具有 patch_h x patch_w 的网格划分
            k = rearrange(self.conv_k(x_high),
                          'b c (h p1) (w p2) -> (b h w) c (p1 p2)',
                          p1=1, p2=1,  # 高层特征的每个"patch"是1x1
                          h=patch_h, w=patch_w)

            # 计算注意力权重
            attn = torch.matmul(q, k)  # (b*patch_h*patch_w, patch_size_h*patch_size_w, 1)
            attn = attn / np.power(self.inter_dim, 0.5)
            attn = self.softmax(attn)

            # 使用key作为value
            v = rearrange(self.conv_k(x_high),
                          'b c (h p1) (w p2) -> (b h w) (p1 p2) c',
                          p1=1, p2=1,
                          h=patch_h, w=patch_w)

            # 应用注意力权重
            output = torch.matmul(attn, v)  # (b*patch_h*patch_w, patch_size_h*patch_size_w, c)

            # 重新排列回原始形状
            output = rearrange(output,
                               '(b h w) (p1 p2) c -> b c (h p1) (w p2)',
                               p1=patch_size_h, p2=patch_size_w,
                               h=patch_h, w=patch_w,
                               b=b)

            return output + x_low

        except Exception as e:
            print(f"SDP error: {e}")
            print(f"x_low: {x_low.shape}, x_high: {x_high.shape}")
            print(f"patch_h: {patch_h}, patch_w: {patch_w}")
            print(f"patch_size_h: {patch_size_h}, patch_size_w: {patch_size_w}")

class SDPv2(nn.Module):
    def __init__(self, dim=256, inter_dim=None):
        super(SDPv2, self).__init__()
        """Improved Version of Spatial Dependency Perception Module SDP"""
        self.inter_dim = inter_dim
        if self.inter_dim == None:
            self.inter_dim = dim
        self.conv_q = nn.Sequential(
            nn.Conv2d(dim, self.inter_dim, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.GroupNorm(32, self.inter_dim))
        self.conv_k = nn.Sequential(
            nn.Conv2d(dim, self.inter_dim, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.GroupNorm(32, self.inter_dim))
        self.conv = nn.Sequential(
            ConvWithoutBN(self.inter_dim, dim, 3, padding=1, bias=False,
                          activation_layer=nn.ReLU(inplace=True)),
            nn.GroupNorm(32, dim)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_low, x_high, patch_size):
        b_, _, h_, w_ = x_low.size()
        q = rearrange(self.conv_q(x_low), 'b c (h p1) (w p2) -> (b h w) c (p1 p2)', p1=patch_size[0], p2=patch_size[1])
        q = q.transpose(1, 2)  # 1,4096,128
        k = rearrange(self.conv_k(x_high), 'b c (h p1) (w p2) -> (b h w) c (p1 p2)', p1=patch_size[0], p2=patch_size[1])
        attn = torch.matmul(q, k)  # 1, 4096, 1024
        attn = attn / np.power(self.inter_dim, 0.5)
        attn = self.softmax(attn)
        v = k.transpose(1, 2)  # 1, 1024, 128
        output = torch.matmul(attn, v)  # 1, 4096, 128
        output = rearrange(output.transpose(1, 2).contiguous(), '(b h w) c (p1 p2) -> b c (h p1) (w p2)',
                           p1=patch_size[0], p2=patch_size[1], h=h_ // patch_size[0], w=w_ // patch_size[1])
        output = self.conv(output + x_low)
        return output



if __name__=="__main__":
    p1_features = torch.randn(2, 256, 160, 160)  # 高分辨率特征图
    p4_features = torch.randn(2, 256, 20, 20)  # 低分辨率特征图

    # P1&P2: 使用DCT高频增强
    dct_p1 = DctSpatialInteraction(256, (0.3, 0.3), isdct=True)
    enhanced_p1 = dct_p1(p1_features)

    # P3&P4: 使用普通空间注意力
    dct_p4 = DctSpatialInteraction(256, (0.3, 0.3), isdct=False)
    enhanced_p4 = dct_p4(p4_features)

    print(enhanced_p1.shape, enhanced_p4.shape)

    p1_input = torch.randn(2, 256, 160, 160)  # P1: 高分辨率
    p2_input = torch.randn(2, 256, 80, 80)  # P2: 高分辨率
    p3_input = torch.randn(2, 256, 40, 40)  # P3: 中分辨率
    p4_input = torch.randn(2, 256, 20, 20)  # P4: 低分辨率

    # 测试不同配置的HFP
    hfp_p1 = HFP(256, ratio=(0.25, 0.25), patch=(16, 16), isdct=True)  # P1: 使用DCT
    hfp_p2 = HFP(256, ratio=(0.25, 0.25), patch=(8, 8), isdct=True)  # P2: 使用DCT
    hfp_p3 = HFP(256, ratio=None, patch=None, isdct=False)  # P3: 不使用DCT
    hfp_p4 = HFP(256, ratio=None, patch=None, isdct=False)  # P4: 不使用DCT

    p1_output = hfp_p1(p1_input)
    p2_output = hfp_p2(p2_input)
    p3_output = hfp_p3(p3_input)
    p4_output = hfp_p4(p4_input)

    print(f"P1输入: {p1_input.shape} -> 输出: {p1_output.shape}")
    print(f"P2输入: {p2_input.shape} -> 输出: {p2_output.shape}")
    print(f"P3输入: {p3_input.shape} -> 输出: {p3_output.shape}")
    print(f"P4输入: {p4_input.shape} -> 输出: {p4_output.shape}")

    sdp_low = torch.randn(1, 256, 32, 32)  # 低层级特征 (如P2)
    sdp_high = torch.randn(1, 256, 16, 16)  # 高层级特征 (如P3)

    sdp = SDP(256)

    sdp_output = sdp([sdp_low, sdp_high])
    print(f"SDP输入: low={sdp_low.shape}, high={sdp_high.shape} -> 输出: {sdp_output.shape}")
