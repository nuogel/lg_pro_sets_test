import torch.nn as nn
import torch

'''
1. 4倍的效果，2倍效果都還可以，單張圖像的PSNR->27.4左右
'''

from ..registry import MODELS


@MODELS.registry()
class SRCNN(nn.Module):
    def __init__(self, cfg):
        super(SRCNN, self).__init__()
        self.cfg = cfg
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU6()
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.relu2 = nn.ReLU6()
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

    def forward(self, input_x,**args):
        x = input_x
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        # out = out.permute(0, 2, 3, 1)

        return out
