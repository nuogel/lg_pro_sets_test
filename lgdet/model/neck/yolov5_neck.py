import torch.nn as nn
import torch
from lgdet.model.common.common_convs import Concat, CBL, BottleneckCSP, DeConv


class YOLOV5NECK(nn.Module):
    def __init__(self, chs=[1024, 512, 256, 256, 256, 512, 512, 1024], csp=[3, 3, 3, 3]):
        super(YOLOV5NECK, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')# trt not support
        self.dconv_upsample1 = DeConv(256, 256)
        self.dconv_upsample2 = DeConv(128, 128)
        self.concat = Concat()
        self.csp01 = BottleneckCSP(chs[0], chs[1], n=csp[0])
        self.cbl0 = CBL(chs[1], chs[2], k=1, s=1)
        self.csp02 = BottleneckCSP(chs[2] * 2, chs[3], n=csp[1])

        self.cbl1 = CBL(chs[3], chs[4], k=3, s=2)
        self.csp1 = BottleneckCSP(chs[4] * 2, chs[5], n=csp[2])

        self.cbl2 = CBL(chs[5], chs[6], k=3, s=2)
        self.csp2 = BottleneckCSP(chs[6] * 2, chs[7], n=csp[3])

    def forward(self, backbone):
        b0, b1, b2 = backbone

        x0 = self.dconv_upsample1(b2)
        # x0 = nn.functional.interpolate(b2, (int(b2.shape[2] * 2), int(b2.shape[3]) * 2), mode='bilinear', align_corners=False)
        x0 = self.concat([x0, b1])
        x0 = self.csp01(x0)
        x0 = self.cbl0(x0)
        x1_0 = x0

        x0 = self.dconv_upsample2(x0)
        # x0 = nn.functional.interpolate(x0, (int(x0.shape[2] * 2), int(x0.shape[3]) * 2), mode='bilinear', align_corners=False)

        x0 = self.concat([x0, b0])
        x0 = self.csp02(x0)
        x1_1 = x0

        x1_1 = self.cbl1(x1_1)
        x1 = self.concat([x1_0, x1_1])
        x1 = self.csp1(x1)
        x2_1 = x1

        x2_1 = self.cbl2(x2_1)
        x2 = self.concat([x2_1, b2])
        x2 = self.csp2(x2)

        return [x0, x1, x2]
