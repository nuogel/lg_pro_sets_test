import torch
import torch.nn as nn
import torch.nn.functional as F
from lgdet.model.backbone.darknet import darknet_tiny1, Conv2d
from ..registry import MODELS


@MODELS.registry()
class YOLOV3_TINY_O(nn.Module):
    def __init__(self, cfg):
        super(YOLOV3_TINY_O, self).__init__()

        self.num_classes = cfg.TRAIN.CLASSES_NUM
        self.anchor_number = cfg.TRAIN.FMAP_ANCHOR_NUM

        # backbone
        self.backbone = darknet_tiny1(pretrained=False)

        # s = 32
        self.conv_set_2 = Conv2d(1024, 256, 3, padding=1, leakyReLU=True)

        self.conv_1x1_2 = Conv2d(256, 128, 1, leakyReLU=True)

        self.extra_conv_2 = Conv2d(256, 512, 3, padding=1, leakyReLU=True)
        self.pred_2 = nn.Conv2d(512, self.anchor_number * (1 + 4 + self.num_classes), 1)

        # s = 16
        self.conv_set_1 = Conv2d(384, 256, 3, padding=1, leakyReLU=True)
        self.pred_1 = nn.Conv2d(256, self.anchor_number * (1 + 4 + self.num_classes), 1)

    def forward(self, input_x,**args):
        x = input_x
        # backbone
        C_4, C_5 = self.backbone(x)

        # detection head
        # multi scale feature map fusion
        C_5 = self.conv_set_2(C_5)
        C_5_up = F.interpolate(self.conv_1x1_2(C_5), scale_factor=2.0, mode='bilinear', align_corners=True)

        C_4 = torch.cat([C_4, C_5_up], dim=1)
        C_4 = self.conv_set_1(C_4)

        # head
        # s = 32
        C_5 = self.extra_conv_2(C_5)
        pred_1 = self.pred_2(C_5)

        # s = 16
        pred_2 = self.pred_1(C_4)

        return [pred_1, pred_2]
