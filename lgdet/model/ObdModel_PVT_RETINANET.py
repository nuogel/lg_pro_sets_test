'''
Model Summary: 269 layers, 11.534817 (M)parameters, 11.533281 (M)gradients, 6.6 GFLOPs
'''

import torch.nn as nn
from ..registry import MODELS
from functools import partial
from lgdet.model.backbone.pvt import PyramidVisionTransformer
from lgdet.model.backbone.pvt_v2 import PyramidVisionTransformerV2
from lgdet.model.ObdModel_RETINANET import RETINANET

@MODELS.registry()
class PVT_RETINANET(RETINANET):
    def __init__(self, config=None):
        super(PVT_RETINANET, self).__init__(config=config)
        pvt = 'PyramidVisionTransformer'  # PyramidVisionTransformerV2
        self.backbone = eval(pvt)(num_stages=4,
                                  patch_size=4, embed_dims=[256, 512, 1024, 2048], num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4],
                                  qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2],
                                  sr_ratios=[8, 4, 2, 1], drop_rate=0.1, drop_path_rate=0.1)

    def forward(self, input_x, **args):
        x = input_x
        _, C3, C4, C5 = self.backbone(x)
        all_p_level = self.fpn([C3, C4, C5])
        cls_logits, reg_preds = self.head(all_p_level)
        anchors = self.anchors(input_x)
        return cls_logits, reg_preds, anchors