from torch import nn
import math
from collections import OrderedDict


class ChannelAttention(nn.Module):
    def __init__(self, num_features, reduction):
        super(ChannelAttention, self).__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.module(x)


class RCAB(nn.Module):
    def __init__(self, num_features, reduction):
        super(RCAB, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            ChannelAttention(num_features, reduction)
        )

    def forward(self, x):
        return x + self.module(x)


class RG(nn.Module):
    def __init__(self, num_features, num_rcab, reduction):
        super(RG, self).__init__()
        self.module = [RCAB(num_features, reduction) for _ in range(num_rcab)]
        self.module.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return x + self.module(x)


def upsampler(scale, num_features):
    upscale = []
    if (scale & (scale - 1)) == 0:
        for i in range(int(math.log(scale, 2))):
            upscale.append(nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1))
            upscale.append(nn.PixelShuffle(2))
    elif scale == 3:
        upscale.append(nn.Conv2d(num_features, 9 * num_features, 3))
        upscale.append(nn.PixelShuffle(3))
    else:
        raise NotImplementedError
    return upscale

from ..registry import MODELS


@MODELS.registry()
class RCAN(nn.Module):
    def __init__(self, cfg):
        super(RCAN, self).__init__()
        scale = cfg.TRAIN.UPSCALE_FACTOR
        num_features = 64  # args.num_features
        num_rg = 3  # 5  # args.num_rg
        num_rcab = 5  # 10  # args.num_rcab
        reduction = 16  # args.reduction
        self.sf = nn.Conv2d(3, num_features, kernel_size=3, padding=1)
        self.rgs = nn.Sequential(*[RG(num_features, num_rcab, reduction) for _ in range(num_rg)])
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.upscale = nn.Sequential(*upsampler(scale, num_features))
        self.conv2 = nn.Conv2d(num_features, 3, kernel_size=3, padding=1)

    def forward(self, input_x,**args):
        x = input_x
        x = self.sf(x)
        residual = x
        x = self.rgs(x)
        x = self.conv1(x)
        x += residual
        x = self.upscale(x)
        x = self.conv2(x)
        # x = x.permute(0, 2, 3, 1)
        return x
