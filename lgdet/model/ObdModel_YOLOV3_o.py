import torch
import torch.nn as nn
import torch.nn.functional as F
from .aid_models.darknet import *
import numpy as np
from ..registry import MODELS


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding=0, stride=1, dilation=1, leakyReLU=False):
        super(Conv2d, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True) if leakyReLU else nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convs(x)


@MODELS.registry()
class YOLOV3(nn.Module):
    def __init__(self, cfg):
        super(YOLOV3, self).__init__()
        device = cfg.TRAIN.DEVICE
        num_classes = cfg.TRAIN.CLASSES_NUM
        hr = False

        self.device = device
        self.num_classes = num_classes
        self.conf_thresh = 0.5
        self.nms_thresh = 0.5
        self.stride = [8, 16, 32]
        self.anchor_number = cfg.TRAIN.FMAP_ANCHOR_NUM

        # backbone darknet-53 (optional: darknet-19)
        self.backbone = darknet53(pretrained=False, hr=hr)

        # s = 32
        self.conv_set_3 = nn.Sequential(
            Conv2d(1024, 512, 1, leakyReLU=True),
            Conv2d(512, 1024, 3, padding=1, leakyReLU=True),
            Conv2d(1024, 512, 1, leakyReLU=True),
            Conv2d(512, 1024, 3, padding=1, leakyReLU=True),
            Conv2d(1024, 512, 1, leakyReLU=True),
        )
        self.conv_1x1_3 = Conv2d(512, 256, 1, leakyReLU=True)
        self.extra_conv_3 = Conv2d(512, 1024, 3, padding=1, leakyReLU=True)
        self.pred_3 = nn.Conv2d(1024, self.anchor_number * (1 + 4 + self.num_classes), 1)

        # s = 16
        self.conv_set_2 = nn.Sequential(
            Conv2d(768, 256, 1, leakyReLU=True),
            Conv2d(256, 512, 3, padding=1, leakyReLU=True),
            Conv2d(512, 256, 1, leakyReLU=True),
            Conv2d(256, 512, 3, padding=1, leakyReLU=True),
            Conv2d(512, 256, 1, leakyReLU=True),
        )
        self.conv_1x1_2 = Conv2d(256, 128, 1, leakyReLU=True)
        self.extra_conv_2 = Conv2d(256, 512, 3, padding=1, leakyReLU=True)
        self.pred_2 = nn.Conv2d(512, self.anchor_number * (1 + 4 + self.num_classes), 1)

        # s = 8
        self.conv_set_1 = nn.Sequential(
            Conv2d(384, 128, 1, leakyReLU=True),
            Conv2d(128, 256, 3, padding=1, leakyReLU=True),
            Conv2d(256, 128, 1, leakyReLU=True),
            Conv2d(128, 256, 3, padding=1, leakyReLU=True),
            Conv2d(256, 128, 1, leakyReLU=True),
        )
        self.extra_conv_1 = Conv2d(128, 256, 3, padding=1, leakyReLU=True)
        self.pred_1 = nn.Conv2d(256, self.anchor_number * (1 + 4 + self.num_classes), 1)


    def forward(self, input_x,**args):
        x = input_x
        # backbone
        fmp_1, fmp_2, fmp_3 = self.backbone(x)

        # detection head
        # multi scale feature map fusion
        fmp_3 = self.conv_set_3(fmp_3)
        fmp_3_up = F.interpolate(self.conv_1x1_3(fmp_3), scale_factor=2.0, mode='bilinear', align_corners=True)

        fmp_2 = torch.cat([fmp_2, fmp_3_up], 1)
        fmp_2 = self.conv_set_2(fmp_2)
        fmp_2_up = F.interpolate(self.conv_1x1_2(fmp_2), scale_factor=2.0, mode='bilinear', align_corners=True)

        fmp_1 = torch.cat([fmp_1, fmp_2_up], 1)
        fmp_1 = self.conv_set_1(fmp_1)

        # head
        # s = 32
        fmp_3 = self.extra_conv_3(fmp_3)
        pred_3 = self.pred_3(fmp_3)

        # s = 16
        fmp_2 = self.extra_conv_2(fmp_2)
        pred_2 = self.pred_2(fmp_2)

        # s = 8
        fmp_1 = self.extra_conv_1(fmp_1)
        pred_1 = self.pred_1(fmp_1)

        preds = [pred_3, pred_2, pred_1]
        return preds
