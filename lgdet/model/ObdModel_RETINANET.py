import torch
import torch.nn as nn
import numpy as np
from lgdet.util.util_anchor_maker import Anchors
from lgdet.model.backbone.resnet import *
from lgdet.model.head.retina_head import RetinaHead
from lgdet.model.neck.fpn_neck import FPN
from ..registry import MODELS


@MODELS.registry()
class RETINANET(nn.Module):
    def __init__(self, config=None):
        super(RETINANET, self).__init__()
        self.config = config
        use_p5 = True
        fpn_out_channels = 256
        self.config.fpn_out_channels = 256
        self.config.use_GN_head = True
        self.config.freeze_bn = False
        self.config.freeze_stage_1 = False
        resnet = 'resnet50'
        expansion_list = {
            'resnet18': 1,
            'resnet34': 1,
            'resnet50': 4,
            'resnet101': 4,
            'resnet152': 4,
        }
        assert resnet in expansion_list

        self.backbone = eval(resnet)(pretrained=True)
        # self.freeze_bn()
        expansion = expansion_list[resnet]
        self.fpn = FPN(channels_of_fetures=[128 * expansion, 256 * expansion, 512 * expansion],features=fpn_out_channels, use_p5=use_p5)
        self.head = RetinaHead(config=self.config)
        self.anchors = Anchors()

    def train(self, mode=True):
        super(RETINANET, self).train(mode=True)

        def freeze_bn(module):
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
            class_name = module.__class__.__name__
            if class_name.find('BatchNorm') != -1:
                for p in module.parameters():
                    p.requires_grad = False

        if self.config.freeze_bn:
            self.apply(freeze_bn)
            print("success freeze bn")
        if self.config.freeze_stage_1:
            self.backbone.freeze_stages(1)
            print("success freeze stage 1")

    def forward(self, input_x, **args):
        x = input_x
        C3, C4, C5 = self.backbone(x)
        all_p_level = self.fpn([C3, C4, C5])
        cls_logits, reg_preds = self.head(all_p_level)
        anchors = self.anchors(input_x)
        # cls_logits = cls_logits.sigmoid()
        return cls_logits, reg_preds, anchors

    def weights_init(self):
        pass
