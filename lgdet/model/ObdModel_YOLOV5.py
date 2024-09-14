"""Yolo v5 net.
Model Summary: 288 layers, 7.307921 (M)parameters, 7.306337 (M)gradients, 2.1 GFLOPs
"""
import torch
import torch.nn as nn
from lgdet.model.backbone.yolov5_backbone import YOLOV5BACKBONE
from lgdet.model.neck.yolov5_neck import YOLOV5NECK
from ..registry import MODELS
import math


@MODELS.registry()
class YOLOV5(nn.Module):
    """Constructs a darknet-21 model.
    """

    def __init__(self, cfg):
        super(YOLOV5, self).__init__()
        self.anc_num = cfg.TRAIN.FMAP_ANCHOR_NUM
        self.cls_num = cfg.TRAIN.CLASSES_NUM
        self.final_out = self.anc_num * (1 + 4 + self.cls_num)
        if cfg.TRAIN.IOU_AWARE:
            self.final_out = self.anc_num * (1 + 4 + 1 + self.cls_num)
        self.layers_out_filters = [64, 128, 256, 512, 1024]

        self.yolov5_type = cfg.TRAIN.TYPE

        if self.yolov5_type == 's':
            backbone_chs = [32, 64, 64, 128, 128, 256, 256, 512, 512, 512, 256]
            backbone_csp = [1, 3, 3, 1]
            neck_chs = [512, 256, 128, 128, 128, 256, 256, 512]
            neck_csp = [1, 1, 1, 1]
            deteck = [128, 256, 512]
        elif self.yolov5_type == 'm':
            ...
        elif self.yolov5_type == 'l':
            backbone_chs = [64, 128, 128, 256, 256, 512, 512, 1024, 1024, 1024, 512]
            backbone_csp = [3, 9, 9, 3]
            neck_chs = [1024, 512, 256, 256, 256, 512, 512, 1024]
            neck_csp = [3, 3, 3, 3]
            deteck = [256, 512, 1024]
        elif self.yolov5_type == 'x':
            ...

        self.backbone = YOLOV5BACKBONE(chs=backbone_chs, csp=backbone_csp)
        self.neck = YOLOV5NECK(chs=neck_chs, csp=neck_csp)
        self.head = nn.ModuleList(nn.Conv2d(x, self.final_out, 1) for x in deteck)

    def forward(self, input_x, **args):
        x = input_x
        backbone = self.backbone(x)
        neck = self.neck(backbone)
        featuremaps = []
        for neck_i, h_i in zip(neck, self.head):
            featuremaps.append(h_i(neck_i))  # conv
        # return featuremaps[::-1]
        return featuremaps

    def weights_init(self):
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        cf = None
        for mi, s in zip(self.head, [8, 16, 32]):  # from
            # mi.weight.data.fill_(0)
            # tricks form yolov5:
            b = mi.bias.view(3, -1)  # conv.bias(255) to (3,85)
            b.data[:, 0] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (self.cls_num - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
