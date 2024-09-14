import torch
import torch.nn as nn
from collections import OrderedDict


class MobileBottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3, stride=1):
        super(MobileBottleneck, self).__init__()
        padding = int((ksize - 1) / 2)
        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=ksize, stride=stride, padding=padding, bias=False,
                               groups=in_ch)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu61 = nn.ReLU6()

        self.conv2 = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu62 = nn.ReLU6()

    def forward(self, input):
        deep_wise = self.relu61(self.bn1(self.conv1(input)))
        point_wise = self.relu62(self.bn2(self.conv2(deep_wise)))
        return point_wise


# def mobilenet(in_ch, out_ch, ksize=3, stride=1):
#     padding = int((ksize - 1) / 2)
#     return nn.Sequential(
#         nn.Conv2d(in_ch, in_ch, kernel_size=ksize, stride=stride, padding=padding, bias=False, groups=in_ch),
#         nn.BatchNorm2d(in_ch),
#         nn.ReLU6(),
#
#         nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
#         nn.BatchNorm2d(out_ch),
#         nn.ReLU6(),
#     )


def make_layer(in_ch, out_ch, conv=False, dw=True, stride=1, ksize=3, max_pool=2, up_samping=0, last_layer=False):
    layers = []
    if conv and dw:
        print('ERROR, conv and dw == TRUE.')
    if last_layer:
        padding = (ksize - 1) // 2
        # the last layer don't need this activation functions.
        layers.append(('conv2d', nn.Conv2d(in_ch, out_ch, kernel_size=ksize, stride=stride, padding=padding)))
    elif dw:
        layers.append(('dw', MobileBottleneck(in_ch, out_ch, ksize=ksize, stride=stride)))
    elif conv:
        padding = (ksize - 1) // 2
        layers.append(('conv2d', nn.Conv2d(in_ch, out_ch, kernel_size=ksize, stride=stride, padding=padding)))
        layers.append(('batchnorm', nn.BatchNorm2d(out_ch)))
        layers.append(('leakyrelu', nn.LeakyReLU(0.1)))

    if max_pool:
        layers.append(('max_pool', nn.MaxPool2d(2, stride=max_pool)))
    if up_samping:
        layers.append(('up_samping', nn.Upsample(scale_factor=up_samping, mode='bilinear', align_corners=True)))

    return nn.Sequential(OrderedDict(layers))


class BackBone(nn.Module):
    def __init__(self):
        super(BackBone, self).__init__()
        self.out_chs = [16, 32, 64, 128, 256, 512, 1024]

        self.layer1 = make_layer(3, self.out_chs[0], conv=True, dw=False)
        self.layer2 = make_layer(self.out_chs[0], self.out_chs[1])
        self.layer3 = make_layer(self.out_chs[1], self.out_chs[2])
        self.layer4 = make_layer(self.out_chs[2], self.out_chs[3])
        self.layer5_conv = make_layer(self.out_chs[3], self.out_chs[4], max_pool=0, up_samping=False)
        self.layer5_pool = make_layer(in_ch=0, out_ch=0, max_pool=2, conv=False, dw=False, up_samping=False)

        self.layer6 = make_layer(self.out_chs[4], self.out_chs[5], max_pool=0)
        self.layer7 = make_layer(self.out_chs[5], self.out_chs[6], max_pool=0)

    def forward(self, input):
        x = self.layer1(input)  # 208
        x = self.layer2(x)  # 104
        x = self.layer3(x)  # 52
        x = self.layer4(x)  # 26
        x = self.layer5_conv(x)  # 26
        f2 = x
        x = self.layer5_pool(x)  # 13
        x = self.layer6(x)
        f1 = self.layer7(x)
        return f1, f2  # f1:13x13x1024; f2:26x26x256

from ..registry import MODELS


@MODELS.registry()
class YOLOV3_TINY_MOBILENET(nn.Module):
    def __init__(self, cfg):
        super(YOLOV3_TINY_MOBILENET, self).__init__()
        if cfg:
            self.anc_num = cfg.TRAIN.FMAP_ANCHOR_NUM
            self.cls_num = cfg.TRAIN.CLASSES_NUM
        else:
            self.anc_num = 3
            self.cls_num = 4
        out_ch = self.anc_num * (1 + 4 + self.cls_num)
        self.ch_1 = [1024, 256, 512, out_ch]
        self.ch_2 = [256, 128, 384, 256, out_ch]

        self.backbone = BackBone()
        self.bb1_1 = make_layer(self.ch_1[0], self.ch_1[1], ksize=1, max_pool=0)
        self.bb1_2 = make_layer(self.ch_1[1], self.ch_1[2], max_pool=0)
        self.bb1_3 = make_layer(self.ch_1[-2], self.ch_1[-1], ksize=1, max_pool=0, last_layer=True)

        self.bb2_1 = make_layer(self.ch_2[0], self.ch_2[1], ksize=1, max_pool=0, up_samping=2)
        self.bb2_cat = torch.cat
        self.bb2_2 = make_layer(self.ch_2[2], self.ch_2[3], max_pool=0)
        self.bb2_3 = make_layer(self.ch_2[-2], self.ch_2[-1], ksize=1, max_pool=0, last_layer=True)

    def forward(self, input_x,**args):
        x = input_x
        f1, f2 = self.backbone(x)  # jump2
        net1 = self.bb1_1(f1)
        net2 = net1  # jump1

        net1 = self.bb1_2(net1)
        net1 = self.bb1_3(net1)  # 13x13x108

        net2 = self.bb2_1(net2)  # jump1_TO   26x26x128
        net2 = self.bb2_cat((f2, net2), 1)  # jump2_TO    26x26x256+26x26x128=26x26x384
        net2 = self.bb2_2(net2)
        net2 = self.bb2_3(net2)
        return [net1, net2]
