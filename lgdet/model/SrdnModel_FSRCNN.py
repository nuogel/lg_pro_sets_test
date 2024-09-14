import torch.nn as nn
import torch

'''
1. 4倍的效果，2倍效果都還可以，單張圖像的PSNR->27.4左右
'''

from ..registry import MODELS


@MODELS.registry()
class FSRCNN(nn.Module):
    def __init__(self, cfg):
        super(FSRCNN, self).__init__()
        self.cfg = cfg
        m = 4
        self.first_part = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2), nn.PReLU())

        self.layers = []
        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=64, out_channels=12, kernel_size=1, stride=1, padding=0), nn.PReLU()))
        for _ in range(m):
            self.layers.append(nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1))
        self.layers.append(nn.PReLU())
        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=12, out_channels=64, kernel_size=1, stride=1, padding=0), nn.PReLU()))

        self.mid_part = torch.nn.Sequential(*self.layers)

        # Deconvolution
        self.last_part = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=9, stride=self.cfg.TRAIN.UPSCALE_FACTOR, padding=4, output_padding=3)

    def forward(self, input_x,**args):
        x = input_x
        out = self.first_part(x)
        out = self.mid_part(out)
        out = self.last_part(out)
        out = out.permute(0, 2, 3, 1)

        return out
