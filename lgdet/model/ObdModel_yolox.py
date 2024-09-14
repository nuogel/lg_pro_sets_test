#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch.nn as nn
import torch
from .head.yolox_head import YOLOXHead
from .backbone.cspdarknet import CSPDarknet
from .neck.yolo_pafpn import YOLOPAFPN
from ..registry import MODELS


@MODELS.registry()
class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, cfg):
        super().__init__()

        self.cls_num = cfg.TRAIN.CLASSES_NUM
        yolox_type = cfg.TRAIN.TYPE
        typedict = {
            '': [0.33, 0.375],
            's': [0.33, 0.5],
            'm': [0.67, 0.75],
            'l': [1, 1]}  # depth = 1.0 width = 1.0
        depth = typedict[yolox_type][0]
        width = typedict[yolox_type][1]
        in_channels = [256, 512, 1024]

        self.backbone = CSPDarknet(dep_mul=depth, wid_mul=width)
        self.neck = YOLOPAFPN(depth=depth, width=width, in_channels=in_channels)
        self.head = YOLOXHead(self.cls_num, width, in_channels=in_channels)

    def forward(self, input_x, **args):
        x = input_x
        backbone = self.backbone(x)
        neck = self.neck(backbone)
        out = self.head(neck)
        for outi in out:
            if torch.isnan(outi.sum()) or outi.sum().sum() == float("inf") or outi.sum().item() == -float("inf"):
                a = 0
        return out
