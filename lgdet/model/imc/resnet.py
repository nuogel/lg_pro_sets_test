from lgdet.model.backbone.resnet import *
import torch.nn as nn

from lgdet.registry import MODELS

@MODELS.registry()
class RESNET(nn.Module):
    def __init__(self, cfg):
        super(RESNET, self).__init__()
        self.cfg = cfg
        modeltype = 'resnet'+str(cfg.TRAIN.TYPE)
        if_include_top = True
        self.model = eval(modeltype)(pretrained=True, num_classes=cfg.TRAIN.CLASSES_NUM, if_include_top=if_include_top)

    def forward(self, input_x, **args):
        x = input_x
        return self.model(x)

    def weights_init(self):
        pass



