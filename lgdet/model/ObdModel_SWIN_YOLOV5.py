"""
Model Summary: 335 layers, 30.717083 (M)parameters, 30.715547 (M)gradients, 9.5 GFLOPs
"""
from lgdet.model.ObdModel_YOLOV5 import YOLOV5
from lgdet.model.backbone.swin_transformer import SwinTransformer
from lgdet.model.common.common_convs import Conv
from ..registry import MODELS



@MODELS.registry()
class SWIN_YOLOV5(YOLOV5):
    """Constructs a darknet-21 model.
    """

    def __init__(self, cfg):
        super(SWIN_YOLOV5, self).__init__(cfg)

        pvt = 'SwinTransformer'  # PyramidVisionTransformerV2
        self.backbone = eval(pvt)(img_size=cfg.TRAIN.IMG_SIZE)
        self.conv0 = Conv(192,128)
        self.conv1 = Conv(384,256)
        self.conv2 = Conv(768,256)

    def forward(self, input_x, **args):
        x = input_x
        backbone = self.backbone(x)[-3:]
        backbone[0]=self.conv0(backbone[0])
        backbone[1]=self.conv1(backbone[1])
        backbone[2]=self.conv2(backbone[2])

        neck = self.neck(backbone)
        featuremaps = []
        for neck_i, h_i in zip(neck, self.head):
            featuremaps.append(h_i(neck_i))  # conv
        # return featuremaps[::-1]
        return featuremaps
