import torch
import torch.nn as nn
import torch.nn.functional as F

class IPFA(nn.Module):
    def __init__(self, in_channels, out_channels, *args):
        super(IPFA, self).__init__()
        # 3x3卷积扩大感受野
        self.conv_3x3_dw = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.conv_1x1_pw = nn.Conv2d(4 * in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        y = self.conv_3x3_dw(x)
        batch_size, channels, height, width = y.shape
        # First, split by channel
        y_0 = y[:, 0::2, :, :]
        y_1 = y[:, 1::2, :, :]
        # Secondly, split by spatial HW
        f000 = y_0[:, :, 0::2, 0::2]  # 偶通道+高偶+宽偶
        f100 = y_0[:, :, 0::2, 1::2]  # 偶通道+高偶+宽奇
        f010 = y_0[:, :, 1::2, 0::2]  # 偶通道+高奇+宽偶
        f110 = y_0[:, :, 1::2, 1::2]  # 偶通道+高奇+宽奇
        f001 = y_1[:, :, 0::2, 0::2]  # 奇通道+高偶+宽偶
        f011 = y_1[:, :, 1::2, 0::2]  # 奇通道+高奇+宽偶
        f101 = y_1[:, :, 0::2, 1::2]  # 奇通道+高偶+宽奇
        f111 = y_1[:, :, 1::2, 1::2]  # 奇通道+高奇+宽奇
        concat_feats = torch.cat([f000, f100, f010, f110, f001, f011, f101, f111], dim=1)
        out = self.conv_1x1_pw(concat_feats)
        return out


class CRC(nn.Module):
    """
    CRC模块: Conv + ReLU + Conv
    """
    def __init__(self, in_channels, out_channels):
        super(CRC, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


class CCSM(nn.Module):
    """
    The Channel Conflict Information Suppression Module
    """
    def __init__(self, in_channels):
        super(CCSM, self).__init__()
        # 输入是三个特征图的拼接，所以通道数是 3 * in_channels
        self.crc_a = CRC(3 * in_channels, 3 * in_channels)
        self.crc_m = CRC(3 * in_channels, 3 * in_channels)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_list):
        concat_feat = torch.cat(x_list, dim=1)  # [B, 3 * C, H, W]
        # 平均池化分支
        x_avg = self.avg_pool(concat_feat)  # [B, 3 * C, 1, 1]
        oca = self.crc_a(x_avg)  # [B, 3 * C, 1, 1]
        # 最大池化分支
        x_max = self.max_pool(concat_feat)  # [B, 3 * C, 1, 1]
        ocm = self.crc_m(x_max)  # [B, 3 * C, 1, 1]

        # 生成通道权重
        wc = self.sigmoid(ocm + oca)  # [B, 3 * C, 1, 1]
        # print(wc.shape, concat_feat.shape)
        oc = concat_feat * wc
        return oc


class SCSM(nn.Module):
    """
    The Spatial Conflict Information Suppression Module
    """
    def __init__(self, in_channels, out_channels=8):
        super(SCSM, self).__init__()
        self.in_channels = in_channels
        # 三个3x3卷积，每个输出 reduction 个通道
        self.conv3_list = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            for _ in range(3)
        ])  # 通道被分为8个组
        # 1x1卷积将通道数降回3
        self.conv1 = nn.Conv2d(3 * out_channels, 3, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_list):
        feat_maps = []
        for i, feat in enumerate(x_list):
            feat_conv = self.conv3_list[i](feat)  # [B, in_channels // group, H, W]
            feat_maps.append(feat_conv)
        concat_feat = torch.cat(feat_maps, dim=1)  # [B, 3 * in_channels, H, W]
        # 1x1卷积生成空间权重
        ws = self.conv1(concat_feat)  # [B, 3, H, W]
        ws = self.softmax(ws)  # 在通道维度归一化
        # 应用空间权重到原始输入特征图
        weighted_feats = []
        for i, feat in enumerate(x_list):
            # 选择对应的权重图并扩展到与特征图相同的通道数
            ws_i = ws[:, i:i + 1, :, :]  # [B, 1, H, W]
            ws_i_expanded = ws_i.expand(-1, self.in_channels, -1, -1)  # [B, C, H, W]
            weighted_feat = feat * ws_i_expanded
            weighted_feats.append(weighted_feat)
        # 输出是加权后的特征图拼接
        os = torch.cat(weighted_feats, dim=1)  # [B, 3*C, H, W]
        return os


class CSFM(nn.Module):
    """
    Conflict Information Suppression Feature Fusion Module
    """
    def __init__(self, c_shallow, c_mid, c_deep, scale_factor=2):
        super(CSFM, self).__init__()
        self.conv_downsample = nn.Conv2d(c_shallow, c_shallow, kernel_size=3, stride=scale_factor, padding=1)
        self.conv_align_shallow = nn.Conv2d(c_shallow, c_mid, kernel_size=1)
        self.conv_align_deep = nn.Conv2d(c_deep, c_mid, kernel_size=1)
        # 两个分支
        self.ccsm = CCSM(c_mid)
        self.scsm = SCSM(c_mid, out_channels=8)
        self.out_conv = nn.Conv2d(c_mid * 3, c_mid, kernel_size=1)

    def forward(self, x: list[torch.Tensor]):
        """
        下采样应该使用步幅卷积，上采样使用双线性插值
        Args:
            x_shallow: Shallow feature map X [B, C1, H1, W1]
            x_mid: Mid feature map Y [B, C, H2, W2]
            x_deep: Deep feature map Z [B, C3, H3, W3]
        """
        x_shallow, x_mid, x_deep = x
        # 调整特征图尺寸到与中层特征图一致
        target_size = x_mid.shape[2:]
        # D 下采样
        x_prime = self.conv_downsample(x_shallow)
        x_prime = self.conv_align_shallow(x_prime)
        # U 上采样
        z_prime = F.interpolate(x_deep, size=target_size, mode='bilinear', align_corners=True)
        z_prime = self.conv_align_deep(z_prime)

        input_list = [x_prime, x_mid, z_prime]
        oc = self.ccsm(input_list)  # [B, 3*C, H, W]\
        os = self.scsm(input_list)  # [B, 3*C, H, W]
        ocs = oc + os  # [B, 3*C, H, W]
        out = self.out_conv(ocs)  # [B, C, H, W]
        return out


if __name__ == "__main__":
    # batch_size, channels, height, width = 4, 64, 32, 32
    # x = torch.randn(batch_size, channels, height, width)
    # ipfa = IPFA(channels, channels)
    # out = ipfa(x)
    # print(f"输入形状: {x.shape}")
    # print(f"输出形状: {out.shape}")

    batch_size, channels = 2, 64
    c_shallow, c_mid, c_deep = 64, 128, 256   # 16, 32, 64  32, 64, 128     64, 128, 256
    x_shallow = torch.randn(batch_size, c_shallow, 32, 32)  # 浅层特征
    x_mid = torch.randn(batch_size, c_mid, 16, 16)    # 中层特征
    x_deep = torch.randn(batch_size, c_deep, 8, 8)  # 深层特征
    # torch.Size([1, 256, 16, 16]) torch.Size([1, 128, 16, 16])
    x2 = torch.randn(batch_size, c_deep, 16, 16)  # 浅层特征
    csfm = CSFM(64, 128, 256)
    output = csfm([x_shallow, x_mid, x_deep])
    print(output.shape, x2.shape)
    output = torch.cat([x2, output], dim=1)
    print(output.shape)
    from ultralytics.nn.modules import Conv
    conv = Conv(384, 512)
    print(conv(output).shape)
    print(f"CSFM 输出形状: {output.shape}")

    # x_shallow = F.interpolate(x_shallow, size=(64, 64), mode='bilinear', align_corners=True)
    # x_deep = F.interpolate(x_deep, size=(64, 64), mode='bilinear', align_corners=True)

    # ccsm = CCSM(channels)
    # output1 = ccsm([x_shallow, x_mid, x_deep])
    # scsm = SCSM(channels, 8)
    # output2 = ccsm([x_shallow, x_mid, x_deep])
    # print(f"输入形状: 浅层{x_shallow.shape}, 中层{x_mid.shape}, 深层{x_deep.shape}")
    # print(f"CCSM 输出形状: {output1.shape}")
    # print(f"SCSM 输出形状: {output2.shape}")








