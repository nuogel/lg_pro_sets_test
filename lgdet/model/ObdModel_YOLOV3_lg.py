"""Yolo v3 net."""
import torch
import torch.nn as nn
from collections import OrderedDict
from lgdet.model.aid_models.DARKNET import DarkNet
from ..registry import MODELS


def darknet_53():
    return DarkNet([1, 2, 8, 8, 4])




@MODELS.registry()
class YOLOV3(nn.Module):
    """Constructs a darknet-21 model.
    """

    def __init__(self, cfg):
        super(YOLOV3, self).__init__()
        self.anc_num = cfg.TRAIN.FMAP_ANCHOR_NUM
        self.cls_num = cfg.TRAIN.CLASSES_NUM
        self.final_out = self.anc_num * (1 + 4 + self.cls_num)
        self.layers_out_filters = [64, 128, 256, 512, 1024]
        self.backbone = darknet_53()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_set0 = self._conv_set('conv_set0', [512, 1024], self.layers_out_filters[-1], self.final_out)
        self.conv_set1 = self._conv_set('conv_set1', [256, 512], self.layers_out_filters[-2] + 256, self.final_out)
        self.conv_set2 = self._conv_set('conv_set2', [128, 256], self.layers_out_filters[-3] + 128, self.final_out)

        self.conv_next1 = self._conv('conv_next1', 512, 256, 1)
        self.conv_next2 = self._conv('conv_next2', 256, 128, 1)

    def _conv(self, name, _in, _out, k_size):
        pad = (k_size - 1) // 2 if k_size else 0
        _set_out = nn.Sequential(OrderedDict([
            (name + 'conv', nn.Conv2d(_in, _out, k_size, stride=1, padding=pad, bias=False)),
            (name + 'bn', nn.BatchNorm2d(_out, momentum=0.9, eps=1e-5)),
            (name + 'relu', nn.LeakyReLU(0.1))
        ]))
        return _set_out

    def _conv_set(self, name, in_out, _in, _out):
        mlist = nn.ModuleList([
            self._conv(name+'1', _in, in_out[0], 1),
            self._conv(name+'2', in_out[0], in_out[1], 3),
            self._conv(name+'3', in_out[1], in_out[0], 1),
            self._conv(name+'4', in_out[0], in_out[1], 3),
            self._conv(name+'5', in_out[1], in_out[0], 1),
            self._conv(name+'6', in_out[0], in_out[1], 3),
        ])
        mlist.add_module(name + "conv_out", nn.Conv2d(in_out[1], _out, 1, 1, 0, True))
        return mlist

    def _branch(self, _embedding, _in):
        out_branch = None
        for i, e in enumerate(_embedding):
            _in = e(_in)
            if i == 4:
                out_branch = _in
        return _in, out_branch

    def forward(self, **args):

        x = args['input_x']
        x2, x1, x0 = self.backbone(x)
        featuremap0, net_out0 = self._branch(self.conv_set0, x0)
        in_x1 = self.conv_next1(net_out0)
        in_x1 = self.upsample(in_x1)
        in_x1 = torch.cat([in_x1, x1], 1)
        featuremap1, net_out1 = self._branch(self.conv_set1, in_x1)
        in_x2 = self.conv_next2(net_out1)
        in_x2 = self.upsample(in_x2)
        in_x2 = torch.cat([in_x2, x2], 1)
        featuremap2, _ = self._branch(self.conv_set2, in_x2)
        return featuremap0, featuremap1, featuremap2
