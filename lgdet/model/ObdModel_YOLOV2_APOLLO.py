"""Definition of the apollo yolo network."""
from collections import OrderedDict
import torch

"""
Definition of the apollo yolo network,
"""
import torch
import torch.nn as nn

from ..registry import MODELS


@MODELS.registry()
class YOLOV2_APOLLO(torch.nn.Module):
    """
    apollo yolo 2d
    """

    def __init__(self, cfg):
        super(YOLOV2_APOLLO, self).__init__()
        self.anc_num = cfg.TRAIN.FMAP_ANCHOR_NUM
        self.cls_num = cfg.TRAIN.CLASSES_NUM
        self.conv1 = torch.nn.Conv2d(3, 16, 3, 1, 1, 1, bias=True)
        self.conv1_relu = torch.nn.ReLU(1)
        self.pool1 = torch.nn.MaxPool2d(2, 2, 0)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, 1, 1, 1, bias=True)
        self.conv2_relu = torch.nn.ReLU(1)
        self.pool2 = torch.nn.MaxPool2d(2, 2, 0)
        self.conv3_1 = torch.nn.Conv2d(32, 64, 3, 1, 1, 1, bias=True)
        self.conv3_1_relu = torch.nn.ReLU(1)
        self.conv3_2 = torch.nn.Conv2d(64, 32, 1, 1, 0, 1, bias=True)
        self.conv3_2_relu = torch.nn.ReLU(1)
        self.conv3_3 = torch.nn.Conv2d(32, 64, 3, 1, 1, 1, bias=True)
        self.conv3_3_relu = torch.nn.ReLU(1)
        self.pool3 = torch.nn.MaxPool2d(2, 2, 0)
        self.conv4_1 = torch.nn.Conv2d(64, 128, 3, 1, 1, 1, bias=True)
        self.conv4_1_relu = torch.nn.ReLU(1)
        self.conv4_2 = torch.nn.Conv2d(128, 64, 1, 1, 0, 1, bias=True)
        self.conv4_2_relu = torch.nn.ReLU(1)
        self.conv4_3 = torch.nn.Conv2d(64, 128, 3, 1, 1, 1, bias=True)
        self.conv4_3_relu = torch.nn.ReLU(1)
        self.pool4 = torch.nn.MaxPool2d(2, 2, 0)
        self.conv5_1 = torch.nn.Conv2d(128, 256, 3, 1, 1, 1, bias=True)
        self.conv5_1_relu = torch.nn.ReLU(1)
        self.conv5_2 = torch.nn.Conv2d(256, 128, 1, 1, 0, 1, bias=True)
        self.conv5_2_relu = torch.nn.ReLU(1)
        self.conv5_3 = torch.nn.Conv2d(128, 256, 3, 1, 1, 1, bias=True)
        self.conv5_3_relu = torch.nn.ReLU(1)
        self.conv5_4 = torch.nn.Conv2d(256, 128, 1, 1, 0, 1, bias=True)
        self.conv5_4_relu = torch.nn.ReLU(1)
        self.conv5_5 = torch.nn.Conv2d(128, 256, 3, 1, 1, 1, bias=True)
        self.conv5_5_relu = torch.nn.ReLU(1)
        self.pool5 = torch.nn.MaxPool2d(3, 1, 1)
        self.conv6_1_nodilate = torch.nn.Conv2d(256, 512, 5, 1, 2, 1, bias=True)
        self.conv6_1_relu = torch.nn.ReLU(1)
        self.conv6_2 = torch.nn.Conv2d(512, 256, 1, 1, 0, 1, bias=True)
        self.conv6_2_relu = torch.nn.ReLU(1)
        self.conv6_3 = torch.nn.Conv2d(256, 512, 3, 1, 1, 1, bias=True)
        self.conv6_3_relu = torch.nn.ReLU(1)
        self.conv6_4 = torch.nn.Conv2d(512, 256, 1, 1, 0, 1, bias=True)
        self.conv6_4_relu = torch.nn.ReLU(1)
        self.conv6_5 = torch.nn.Conv2d(256, 512, 3, 1, 1, 1, bias=True)
        self.conv6_5_relu = torch.nn.ReLU(1)
        self.conv7_1 = torch.nn.Conv2d(512, 512, 3, 1, 1, 1, bias=True)
        self.conv7_1_relu = torch.nn.ReLU(1)
        self.conv7_2 = torch.nn.Conv2d(512, 512, 3, 1, 1, 1, bias=True)
        self.conv7_2_relu = torch.nn.ReLU(1)
        self.concat8 = torch.cat
        self.conv9 = torch.nn.Conv2d(768, 512, 3, 1, 1, 1, bias=True)
        self.conv9_relu = torch.nn.ReLU(1)
        self.conv_final = torch.nn.Conv2d(512, self.anc_num * (4 + 1 + self.cls_num), 1, 1, 0, bias=True)

    def forward(self, input_x,**args):
        x = input_x
        x = self.conv1_relu(self.conv1(x))
        x = self.pool1(x)
        x = self.conv2_relu(self.conv2(x))
        x = self.pool2(x)
        x = self.conv3_1_relu(self.conv3_1(x))
        x = self.conv3_2_relu(self.conv3_2(x))
        x = self.conv3_3_relu(self.conv3_3(x))
        x = self.pool3(x)
        x = self.conv4_1_relu(self.conv4_1(x))
        x = self.conv4_2_relu(self.conv4_2(x))
        x = self.conv4_3_relu(self.conv4_3(x))
        x = self.pool4(x)
        x = self.conv5_1_relu(self.conv5_1(x))
        x = self.conv5_2_relu(self.conv5_2(x))
        x = self.conv5_3_relu(self.conv5_3(x))
        x = self.conv5_4_relu(self.conv5_4(x))
        cat = self.conv5_5_relu(self.conv5_5(x))
        x = self.pool5(cat)
        x = self.conv6_1_relu(self.conv6_1_nodilate(x))
        x = self.conv6_2_relu(self.conv6_2(x))
        x = self.conv6_3_relu(self.conv6_3(x))
        x = self.conv6_4_relu(self.conv6_4(x))
        x = self.conv6_5_relu(self.conv6_5(x))
        x = self.conv7_1_relu(self.conv7_1(x))
        x = self.conv7_2_relu(self.conv7_2(x))
        x = self.concat8((cat, x), 1)
        x = self.conv9_relu(self.conv9(x))
        x = self.conv_final(x)
        return [x, ]