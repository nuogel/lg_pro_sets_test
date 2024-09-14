import torch
import torch.nn as nn
from collections import OrderedDict


def deep_point_wise(in_ch, out_ch, ksize=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, kernel_size=ksize, stride=stride, padding=1, bias=False, groups=in_ch),
        nn.BatchNorm2d(in_ch),
        nn.ReLU6(),

        nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU6(),
    )


def make_layer(in_ch, out_ch, ksize=3, conv_stride=1, max_pool=2, conv=True, up_samping=0, last_layer=False):
    layers = []
    if conv:
        padding = (ksize - 1) // 2
        layers.append(('conv2d', nn.Conv2d(in_ch, out_ch, kernel_size=ksize, stride=conv_stride, padding=padding)))
        if not last_layer:  # the last layer don't need this activation functions.
            layers.append(('batchnorm', nn.BatchNorm2d(out_ch)))
            layers.append(('leakyrelu', nn.LeakyReLU(0.1)))
    if max_pool:
        layers.append(('max_pool', nn.MaxPool2d(2, stride=max_pool)))
    if up_samping:
        layers.append(('up_samping', nn.Upsample(scale_factor=up_samping, mode='bilinear', align_corners=True)))

    return nn.Sequential(OrderedDict(layers))


class Conv_DW(nn.Module):
    def __init__(self, in_ch, out_ch, conv_strid, dw_stride):
        super(Conv_DW, self).__init__()
        self.conv = make_layer(in_ch, out_ch, ksize=1, conv_stride=conv_strid, max_pool=0)
        self.conv_dw = deep_point_wise(out_ch, out_ch, stride=dw_stride)

    def forward(self, input):
        x = self.conv(input)
        x = self.conv_dw(x)
        return x


class BackBone(nn.Module):
    def __init__(self):
        super(BackBone, self).__init__()
        self.out_chs = [3, 32, 64, 128, 256, 512, 1024]

        self.layer1 = Conv_DW(self.out_chs[0], self.out_chs[1], conv_strid=2, dw_stride=1)  # ch:3->32
        self.layer2 = Conv_DW(self.out_chs[1], self.out_chs[2], conv_strid=1, dw_stride=2)  # ch:32->64
        self.layer3 = Conv_DW(self.out_chs[2], self.out_chs[3], conv_strid=1, dw_stride=1)  # ch:64->128
        self.layer4 = Conv_DW(self.out_chs[3], self.out_chs[3], conv_strid=1, dw_stride=2)  # ch:128->128
        self.layer5 = Conv_DW(self.out_chs[3], self.out_chs[4], conv_strid=1, dw_stride=1)  # ch:128->256
        self.layer6 = Conv_DW(self.out_chs[4], self.out_chs[4], conv_strid=1, dw_stride=2)  # ch:256->256
        # self.layer5 =====> feature map 2
        self.layer7_0 = Conv_DW(self.out_chs[4], self.out_chs[5], conv_strid=1, dw_stride=1)  # ch:256->512
        self.layer7_1 = Conv_DW(self.out_chs[5], self.out_chs[5], conv_strid=1, dw_stride=1)  # ch:512->512
        # self.layer7_2 = Conv_DW(self.out_chs[5], self.out_chs[5], conv_strid=1, dw_stride=1)  # ch:512->512
        # self.layer7_3 = Conv_DW(self.out_chs[5], self.out_chs[5], conv_strid=1, dw_stride=1)  # ch:512->512
        # self.layer7_4 = Conv_DW(self.out_chs[5], self.out_chs[5], conv_strid=1, dw_stride=1)  # ch:512->512
        # self.layer7_5 = Conv_DW(self.out_chs[5], self.out_chs[5], conv_strid=1, dw_stride=1)  # ch:512->512
        self.layer7_6 = Conv_DW(self.out_chs[5], self.out_chs[5], conv_strid=1, dw_stride=2)  # ch:512->512

        self.layer8 = Conv_DW(self.out_chs[5], self.out_chs[6], conv_strid=1, dw_stride=1)  # ch:512->512
        self.layer9 = make_layer(self.out_chs[6], self.out_chs[6], max_pool=0)
        # self.layer9 =====> feature map 1

    def forward(self, input):
        x = self.layer1(input)  # 208
        x = self.layer2(x)  # 104
        x = self.layer3(x)  # 52
        x = self.layer4(x)  # 26
        x = self.layer5(x)  # 26
        x = self.layer6(x)  # 26
        f2 = x

        x = self.layer7_0(x)  # 13
        x = self.layer7_1(x)
        # x = self.layer7_2(x)
        # x = self.layer7_3(x)
        # x = self.layer7_4(x)
        # x = self.layer7_5(x)
        x = self.layer7_6(x)

        x = self.layer8(x)
        f1 = self.layer9(x)

        return f1, f2  # f1:13x13x1024; f2:26x26x256

from ..registry import MODELS


@MODELS.registry()
class YOLOV3_MOBILENET(nn.Module):
    def __init__(self, cfg):
        super(YOLOV3_MOBILENET, self).__init__()
        self.anc_num = cfg.TRAIN.FMAP_ANCHOR_NUM
        self.cls_num = cfg.TRAIN.CLASSES_NUM
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
        f_map1 = self.bb1_3(net1)  # 13x13x108

        net2 = self.bb2_1(net2)  # jump1_TO   26x26x128
        net2 = self.bb2_cat((f2, net2), 1)  # jump2_TO    26x26x256+26x26x128=26x26x384
        net2 = self.bb2_2(net2)
        f_map2 = self.bb2_3(net2)

        return [f_map1, f_map2, ]
