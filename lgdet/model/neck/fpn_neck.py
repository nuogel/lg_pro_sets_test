import torch.nn as nn
import torch.nn.functional as F
import math


class FPN_1(nn.Module):
    def __init__(self, channels_of_fetures, channel_out=256):
        """
        fpn,特征金字塔
        :param channels_of_fetures: list,输入层的通道数,必须与输入特征图相对应
        :param channel_out:
        """
        super(FPN_1, self).__init__()
        self.channels_of_fetures = channels_of_fetures

        self.lateral_conv1 = nn.Conv2d(channels_of_fetures[2], channel_out, kernel_size=1, stride=1, padding=0)
        self.lateral_conv2 = nn.Conv2d(channels_of_fetures[1], channel_out, kernel_size=1, stride=1, padding=0)
        self.lateral_conv3 = nn.Conv2d(channels_of_fetures[0], channel_out, kernel_size=1, stride=1, padding=0)

        self.top_down_conv1 = nn.Conv2d(channel_out, channel_out, kernel_size=3, stride=1, padding=1)
        self.top_down_conv2 = nn.Conv2d(channel_out, channel_out, kernel_size=3, stride=1, padding=1)
        self.top_down_conv3 = nn.Conv2d(channel_out, channel_out, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        """

        :param features:
        :return:
        """
        c3, c4, c5 = features

        p5 = self.lateral_conv1(c5)  # 19
        p4 = self.lateral_conv2(c4)  # 38
        p3 = self.lateral_conv3(c3)  # 75

        p4 = F.interpolate(input=p5, size=(p4.size(2), p4.size(3)), mode="nearest") + p4
        p3 = F.interpolate(input=p4, size=(p3.size(2), p3.size(3)), mode="nearest") + p3

        p5 = self.top_down_conv1(p5)
        p4 = self.top_down_conv1(p4)
        p3 = self.top_down_conv1(p3)

        return p3, p4, p5


class FPN(nn.Module):
    '''only for resnet50,101,152'''

    def __init__(self, channels_of_fetures, features=256, use_p5=True):
        super(FPN, self).__init__()
        self.prj_5 = nn.Conv2d(channels_of_fetures[2], features, kernel_size=1)
        self.prj_4 = nn.Conv2d(channels_of_fetures[1], features, kernel_size=1)
        self.prj_3 = nn.Conv2d(channels_of_fetures[0], features, kernel_size=1)
        self.conv_5 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv_4 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv_3 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        if use_p5:
            self.conv_out6 = nn.Conv2d(features, features, kernel_size=3, padding=1, stride=2)
        else:
            self.conv_out6 = nn.Conv2d(256, features, kernel_size=3, padding=1, stride=2)
        self.conv_out7 = nn.Conv2d(features, features, kernel_size=3, padding=1, stride=2)
        self.use_p5 = use_p5
        self.apply(self.init_conv_kaiming)

    def upsamplelike(self, inputs):
        src, target = inputs
        return F.interpolate(src, size=(target.shape[2], target.shape[3]),
                             mode='nearest')

    def init_conv_kaiming(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, a=1)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        C3, C4, C5 = x
        P5 = self.prj_5(C5)
        P4 = self.prj_4(C4)
        P3 = self.prj_3(C3)

        P4 = P4 + self.upsamplelike([P5, C4])
        P3 = P3 + self.upsamplelike([P4, C3])

        P3 = self.conv_3(P3)
        P4 = self.conv_4(P4)
        P5 = self.conv_5(P5)

        P5 = P5 if self.use_p5 else C5
        P6 = self.conv_out6(P5)
        P7 = self.conv_out7(F.relu(P6))
        return [P3, P4, P5, P6, P7]
