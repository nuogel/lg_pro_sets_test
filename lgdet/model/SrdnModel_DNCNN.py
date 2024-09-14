import torch.nn as nn

'''
相比 CBDNET 训练慢许多
'''

from ..registry import MODELS


@MODELS.registry()
class DnCNN(nn.Module):
    def __init__(self, cfg):
        channels = 3
        num_of_layers = 17
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, input_x,**args):
        x = input_x
        out = self.dncnn(x)
        out = out.permute([0, 2, 3, 1])
        '''
        out: represent the noise of the noise image.not the clean image.
        '''
        return out
