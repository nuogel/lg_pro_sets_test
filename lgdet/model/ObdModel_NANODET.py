"""Yolo v3 net."""
import torch
import torch.nn as nn
from lgdet.model.backbone.shufflenetv2 import ShuffleNetV2
from lgdet.model.neck.ghost_pan import GhostPAN
from lgdet.model.head.nanodet_plus_head import NanoDetPlusHead
from ..registry import MODELS
import math


@MODELS.registry()
class NANODET(nn.Module):
    """nanodet plus
    """

    def __init__(self, cfg):
        super(NANODET, self).__init__()
        self.anc_num = cfg.TRAIN.FMAP_ANCHOR_NUM
        self.cls_num = cfg.TRAIN.CLASSES_NUM
        self.final_out = self.anc_num * (1 + 4 + self.cls_num)

        self.backbone = ShuffleNetV2(cfg)

        self.neck = GhostPAN(in_channels=[116, 232, 464],
                             out_channels=96,
                             kernel_size=5,
                             num_extra_level=1,
                             use_depthwise=True,
                             activation='LeakyReLU')
        self.head = NanoDetPlusHead(
            num_classes=self.cls_num,
            input_channel=96,
            feat_channels=96,
            stacked_convs=2,
            kernel_size=5,
            strides=[8, 16, 32, 64],
            activation='LeakyReLU',
            reg_max=7)

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
