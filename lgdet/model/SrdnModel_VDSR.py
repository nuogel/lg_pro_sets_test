import torch.nn as nn
import torch

from ..registry import MODELS


@MODELS.registry()
class VDSR(nn.Module):
    '''
    VDSR: 因為加了全局殘差，因此速度非常快。
    '''
    def __init__(self, cfg):
        super(VDSR, self).__init__()
        self.cfg = cfg
        num_channels = 3
        base_channels = 64
        num_residuals = 4
        self.input_conv = nn.Sequential(nn.Conv2d(num_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(inplace=True))
        self.residual_layers = nn.Sequential(
            *[nn.Sequential(nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(inplace=True)) for _ in range(num_residuals)])
        self.output_conv = nn.Conv2d(base_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, input_x,**args):
        x = input_x
        residual = x
        x = self.input_conv(x)
        x = self.residual_layers(x)
        x = self.output_conv(x)
        x = torch.add(x, residual)
        # x = x.permute(0, 2, 3, 1)

        return x
