import torch.nn as nn
import torch

from ..registry import MODELS


@MODELS.registry()
class ESPCN(nn.Module):
    def __init__(self, cfg):
        super(ESPCN, self).__init__()
        self.cfg = cfg
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 3 * self.cfg.TRAIN.UPSCALE_FACTOR ** 2, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(self.cfg.TRAIN.UPSCALE_FACTOR)

    def forward(self, input_x,**args):
        x = input_x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.pixel_shuffle(x)
        x = x.permute(0, 2, 3, 1)

        return x
