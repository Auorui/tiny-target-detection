import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv

class MKP(nn.Module):
    def __init__(self, dim):
        super(MKP, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.conv2 = Conv(dim, dim, k=1, s=1)
        self.conv3 = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.conv4 = Conv(dim, dim, k=1, s=1)
        self.conv5 = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = x5 + x
        return x6

class FBRTDown(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv2 = Conv(dim_in, dim_in, 3, 2, 1, g=dim_in // 2, act=False)
        self.conv4 = Conv(dim_in, dim_out, 1, 1)

    def forward(self, x):
        x = self.conv2(x)
        x = self.conv4(x)
        return x


class Channel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, 3,
            1, 1, groups=dim
        )
        self.Apt = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x2 = self.dwconv(x)
        x5 = self.Apt(x2)
        x6 = self.sigmoid(x5)

        return x6

class Spatial(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, 1, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(x)
        x5 = self.bn(x1)
        x6 = self.sigmoid(x5)

        return x6

class FCM(nn.Module):
    def __init__(self, dim, alpha=0.75, use_fusion=False):
        super().__init__()
        self.one = int(dim * alpha)      # αC channel (default 75%)
        self.two = dim - self.one        # (1-α)C channel (default 25%)
        # Semantic information branch
        self.semantic_branch = nn.Sequential(
            Conv(self.one, self.one, 3, 1, 1),
            Conv(self.one, self.one, 3, 1, 1),
            Conv(self.one, dim, 1, 1)
        )
        # spatial information branch
        self.spatial_branch = Conv(self.two, dim, 1, 1)
        # Attention mechanism
        self.channel_attention = Channel(dim)
        self.spatial_attention = Spatial(dim)
        # optional final fusion layer
        self.fusion_conv = Conv(dim, dim, 1, 1)
        self.use_fusion = use_fusion

    def forward(self, x):
        x_semantic, x_spatial = torch.split(x, [self.one, self.two], dim=1)
        x_c = self.semantic_branch(x_semantic)  # X^C ∈ R^{C×H×W}
        x_s = self.spatial_branch(x_spatial)  # X^S ∈ R^{C×H×W}
        channel_weights = self.channel_attention(x_c)  # ω1 ∈ R^{C×1×1}
        enhanced_spatial = x_s * channel_weights  # X^S ⊗ ω1
        spatial_weights = self.spatial_attention(x_s)  # ω2 ∈ R^{1×H×W}
        enhanced_semantic = x_c * spatial_weights  # X^C ⊗ ω2
        x_fcm = enhanced_semantic + enhanced_spatial  # (X^C⊗ω2) ⊕ (X^S⊗ω1)
        if self.use_fusion:
            x_fcm = self.fusion_conv(x_fcm)
        return x_fcm

if __name__=="__main__":
    mkp = MKP(dim=64)
    x = torch.randn(2, 64, 32, 32)
    print(x.shape)
    output = mkp(x)
    print(output.shape)
    fcm = FCM(64)
    output_fcm = fcm(x)
    print(output_fcm.shape)