import torch.nn as nn
from lgdet.model.head.fcos_head import ClsCntRegHead
from lgdet.model.neck.fpn_neck import FPN
from lgdet.model.backbone.resnet import resnet50

from ..registry import MODELS


@MODELS.registry()
class FCOS(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.num_classes = config.TRAIN.CLASSES_NUM
        self.backbone = resnet50(pretrained=config.pretrained, if_include_top=False)
        self.fpn = FPN(config.fpn_out_channels, use_p5=config.use_p5)
        self.head = ClsCntRegHead(config.fpn_out_channels, self.num_classes,
                                  config.use_GN_head, config.cnt_on_reg, config.prior)
        self.config = config

    def train(self, mode=True):
        '''
        set module training mode, and frozen bn
        '''
        super().train(mode=True)

        def freeze_bn(module):
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
            classname = module.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in module.parameters(): p.requires_grad = False

        if self.config.freeze_bn:
            self.apply(freeze_bn)
            print("success frozen BN")
        if self.config.freeze_stage_1:
            self.backbone.freeze_stages(1)
            print("success frozen backbone stage1")

    def weights_init(self):
        pass

    def forward(self, input_x, **args):
        x = input_x
        C3, C4, C5 = self.backbone(x)
        all_P = self.fpn([C3, C4, C5])
        cls_logits, cnt_logits, reg_preds = self.head(all_P)
        return [cls_logits, cnt_logits, reg_preds]
