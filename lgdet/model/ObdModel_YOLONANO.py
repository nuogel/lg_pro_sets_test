import torch
import torch.nn as nn
import torch.nn.functional as F


def conv1x1(input_channels, output_channels, stride=1, bn=True):
    # 1x1 convolution without padding
    if bn == True:
        return nn.Sequential(
            nn.Conv2d(
                input_channels, output_channels, kernel_size=1,
                stride=stride, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU6(inplace=True)
        )
    else:
        return nn.Conv2d(
            input_channels, output_channels, kernel_size=1,
            stride=stride, bias=False)


def conv3x3(input_channels, output_channels, stride=1, bn=True):
    # 3x3 convolution with padding=1
    if bn == True:
        return nn.Sequential(
            nn.Conv2d(
                input_channels, output_channels, kernel_size=3,
                stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU6(inplace=True)
        )
    else:
        nn.Conv2d(
            input_channels, output_channels, kernel_size=3,
            stride=stride, padding=1, bias=False)


def sepconv3x3(input_channels, output_channels, stride=1, expand_ratio=1):
    return nn.Sequential(
        # pw
        nn.Conv2d(
            input_channels, input_channels * expand_ratio,
            kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(input_channels * expand_ratio),
        nn.ReLU6(inplace=True),
        # dw
        nn.Conv2d(
            input_channels * expand_ratio, input_channels * expand_ratio, kernel_size=3,
            stride=stride, padding=1, groups=input_channels * expand_ratio, bias=False),
        nn.BatchNorm2d(input_channels * expand_ratio),
        nn.ReLU6(inplace=True),
        # pw-linear
        nn.Conv2d(
            input_channels * expand_ratio, output_channels,
            kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(output_channels)
    )


class EP(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super(EP, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.use_res_connect = self.stride == 1 and input_channels == output_channels

        self.sepconv = sepconv3x3(input_channels, output_channels, stride=stride)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.sepconv(x)

        return self.sepconv(x)


class PEP(nn.Module):
    def __init__(self, input_channels, output_channels, x, stride=1):
        super(PEP, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.use_res_connect = self.stride == 1 and input_channels == output_channels

        self.conv = conv1x1(input_channels, x)
        self.sepconv = sepconv3x3(x, output_channels, stride=stride)

    def forward(self, x):
        out = self.conv(x)
        out = self.sepconv(out)
        if self.use_res_connect:
            return out + x

        return out


class FCA(nn.Module):
    def __init__(self, channels, reduction_ratio):
        super(FCA, self).__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio

        hidden_channels = channels // reduction_ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_channels, bias=False),
            nn.ReLU6(inplace=True),
            nn.Linear(hidden_channels, channels, bias=False),
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        out = self.avg_pool(x).view(b, c)
        out = self.fc(out).view(b, c, 1, 1)
        out = x * out.expand_as(x)
        return out

from ..registry import MODELS


@MODELS.registry()
class YOLONANO(nn.Module):
    def __init__(self, cfg):
        super(YOLONANO, self).__init__()
        self.num_classes = cfg.TRAIN.CLASSES_NUM
        self.num_anchors = cfg.TRAIN.FMAP_ANCHOR_NUM
        self.yolo_channels = (self.num_classes + 5) * self.num_anchors

        # image:  416x416x3
        self.conv1 = conv3x3(3, 12, stride=1)  # output: 416x416x12
        self.conv2 = conv3x3(12, 24, stride=2)  # output: 208x208x24
        self.pep1 = PEP(24, 24, 7, stride=1)  # output: 208x208x24
        self.ep1 = EP(24, 70, stride=2)  # output: 104x104x70
        self.pep2 = PEP(70, 70, 25, stride=1)  # output: 104x104x70
        self.pep3 = PEP(70, 70, 24, stride=1)  # output: 104x104x70
        self.ep2 = EP(70, 150, stride=2)  # output: 52x52x150
        self.pep4 = PEP(150, 150, 56, stride=1)  # output: 52x52x150
        self.conv3 = conv1x1(150, 150, stride=1)  # output: 52x52x150
        self.fca1 = FCA(150, 8)  # output: 52x52x150
        self.pep5 = PEP(150, 150, 73, stride=1)  # output: 52x52x150
        self.pep6 = PEP(150, 150, 71, stride=1)  # output: 52x52x150

        self.pep7 = PEP(150, 150, 75, stride=1)  # output: 52x52x150
        self.ep3 = EP(150, 325, stride=2)  # output: 26x26x325
        self.pep8 = PEP(325, 325, 132, stride=1)  # output: 26x26x325
        self.pep9 = PEP(325, 325, 124, stride=1)  # output: 26x26x325
        self.pep10 = PEP(325, 325, 141, stride=1)  # output: 26x26x325
        self.pep11 = PEP(325, 325, 140, stride=1)  # output: 26x26x325
        self.pep12 = PEP(325, 325, 137, stride=1)  # output: 26x26x325
        self.pep13 = PEP(325, 325, 135, stride=1)  # output: 26x26x325
        self.pep14 = PEP(325, 325, 133, stride=1)  # output: 26x26x325

        self.pep15 = PEP(325, 325, 140, stride=1)  # output: 26x26x325
        self.ep4 = EP(325, 545, stride=2)  # output: 13x13x545
        self.pep16 = PEP(545, 545, 276, stride=1)  # output: 13x13x545
        self.conv4 = conv1x1(545, 230, stride=1)  # output: 13x13x230
        self.ep5 = EP(230, 489, stride=1)  # output: 13x13x489
        self.pep17 = PEP(489, 469, 213, stride=1)  # output: 13x13x469

        self.conv5 = conv1x1(469, 189, stride=1)  # output: 13x13x189
        self.conv6 = conv1x1(189, 105, stride=1)  # output: 13x13x105
        # upsampling conv6 to 26x26x105
        # concatenating [conv6, pep15] -> pep18 (26x26x430)
        self.pep18 = PEP(430, 325, 113, stride=1)  # output: 26x26x325
        self.pep19 = PEP(325, 207, 99, stride=1)  # output: 26x26x325

        self.conv7 = conv1x1(207, 98, stride=1)  # output: 26x26x98
        self.conv8 = conv1x1(98, 47, stride=1)  # output: 26x26x47
        # upsampling conv8 to 52x52x47
        # concatenating [conv8, pep7] -> pep20 (52x52x197)
        self.pep20 = PEP(197, 122, 58, stride=1)  # output: 52x52x122
        self.pep21 = PEP(122, 87, 52, stride=1)  # output: 52x52x87
        self.pep22 = PEP(87, 93, 47, stride=1)  # output: 52x52x93
        self.conv9 = conv1x1(93, self.yolo_channels, stride=1, bn=False)  # output: 52x52x yolo_channels

        # conv7 -> ep6
        self.ep6 = EP(98, 183, stride=1)  # output: 26x26x183
        self.conv10 = conv1x1(183, self.yolo_channels, stride=1, bn=False)  # output: 26x26x yolo_channels

        # conv5 -> ep7
        self.ep7 = EP(189, 462, stride=1)  # output: 13x13x462
        self.conv11 = conv1x1(462, self.yolo_channels, stride=1, bn=False)  # output: 13x13x yolo_channels

    def forward(self, input_x,**args):
        x = input_x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.pep1(out)
        out = self.ep1(out)
        out = self.pep2(out)
        out = self.pep3(out)
        out = self.ep2(out)
        out = self.pep4(out)
        out = self.conv3(out)
        out = self.fca1(out)
        out = self.pep5(out)
        out = self.pep6(out)

        out_pep7 = self.pep7(out)
        out = self.ep3(out_pep7)
        out = self.pep8(out)
        out = self.pep9(out)
        out = self.pep10(out)
        out = self.pep11(out)
        out = self.pep12(out)
        out = self.pep13(out)
        out = self.pep14(out)

        out_pep15 = self.pep15(out)
        out = self.ep4(out_pep15)
        out = self.pep16(out)
        out = self.conv4(out)
        out = self.ep5(out)
        out = self.pep17(out)

        out_conv5 = self.conv5(out)
        out = F.interpolate(self.conv6(out_conv5), scale_factor=2)
        out = torch.cat([out, out_pep15], dim=1)
        out = self.pep18(out)
        out = self.pep19(out)

        out_conv7 = self.conv7(out)
        out = F.interpolate(self.conv8(out_conv7), scale_factor=2)
        out = torch.cat([out, out_pep7], dim=1)
        out = self.pep20(out)
        out = self.pep21(out)
        out = self.pep22(out)
        featuremap0 = self.conv9(out)

        out = self.ep6(out_conv7)
        featuremap1 = self.conv10(out)

        out = self.ep7(out_conv5)
        featuremap2 = self.conv11(out)

        return featuremap0, featuremap1, featuremap2
