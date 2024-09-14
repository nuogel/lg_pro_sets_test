import torch.nn as nn
from lgdet.model.common.common_convs import FocusConvLG as Focus, CBL, BottleneckCSP, SPP, Conv


class YOLOV5BACKBONE(nn.Module):
    def __init__(self, chs=[32, 64, 64, 128, 128, 256, 256, 512, 512, 512, 256], csp=[1, 3, 3, 1]):
        super(YOLOV5BACKBONE, self).__init__()
        self.m = nn.Sequential(
            Focus(3, chs[0], k=3),
            CBL(chs[0], chs[1], k=3, s=2),
            BottleneckCSP(chs[1], chs[2], n=csp[0]),
            CBL(chs[2], chs[3], k=3, s=2),
            BottleneckCSP(chs[3], chs[4], n=csp[1]),  # concat
            CBL(chs[4], chs[5], k=3, s=2),
            BottleneckCSP(chs[5], chs[6], n=csp[2]),  # concat
            CBL(chs[6], chs[7], k=3, s=2),
            SPP(chs[7], chs[8], k=[5, 9, 13]),
            BottleneckCSP(chs[8], chs[9], n=csp[3], shortcut=False),
            CBL(chs[9], chs[10], k=1, s=1)  # concat
        )

    def forward(self, x):
        outs = []
        output_id = [4, 6, 10]
        for i, m in enumerate(self.m):
            x = m(x)
            if i in output_id:
                outs.append(x)
        return outs
