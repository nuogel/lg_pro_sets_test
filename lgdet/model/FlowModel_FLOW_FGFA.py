import torch.nn as nn
import torch


def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, 1:target.size(2) + 1, 1:target.size(3) + 1]


from ..registry import MODELS


@MODELS.registry()
class FLOW_FGFA(nn.Module):
    def __init__(self, cfg=None):
        super(FLOW_FGFA, self).__init__()

        self.flow_conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.conv3_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv4_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.conv6_1 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)

        self.Convolution1 = nn.Conv2d(1024, 2, kernel_size=3, stride=1, padding=1)
        self.Convolution2 = nn.Conv2d(1026, 2, kernel_size=3, stride=1, padding=1)
        self.Convolution3 = nn.Conv2d(770, 2, kernel_size=3, stride=1, padding=1)
        self.Convolution4 = nn.Conv2d(386, 2, kernel_size=3, stride=1, padding=1)
        self.Convolution5 = nn.Conv2d(194, 2, kernel_size=3, stride=1, padding=1)

        self.deconv5 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2)
        self.deconv4 = nn.ConvTranspose2d(1026, 256, kernel_size=4, stride=2)
        self.deconv3 = nn.ConvTranspose2d(770, 128, kernel_size=4, stride=2)
        self.deconv2 = nn.ConvTranspose2d(386, 64, kernel_size=4, stride=2)

        self.upsample_flow6to5 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2)
        self.upsample_flow5to4 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2)
        self.upsample_flow4to3 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2)
        self.upsample_flow3to2 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2)

        self.relu = nn.LeakyReLU(0.1, inplace=True)

        self.avgpool = nn.AvgPool2d(2, stride=2, ceil_mode=True)

    def forward(self, input_x, **args):
        x = input_x
        x = self.avgpool(x)
        conv1 = self.flow_conv1(x)
        relu1 = self.relu(conv1)
        conv2 = self.conv2(relu1)
        relu2 = self.relu(conv2)
        conv3 = self.conv3(relu2)
        relu3 = self.relu(conv3)
        conv3_1 = self.conv3_1(relu3)
        relu4 = self.relu(conv3_1)
        conv4 = self.conv4(relu4)
        relu5 = self.relu(conv4)
        conv4_1 = self.conv4_1(relu5)
        relu6 = self.relu(conv4_1)
        conv5 = self.conv5(relu6)
        relu7 = self.relu(conv5)
        conv5_1 = self.conv5_1(relu7)
        relu8 = self.relu(conv5_1)
        conv6 = self.conv6(relu8)
        relu9 = self.relu(conv6)
        conv6_1 = self.conv6_1(relu9)
        relu10 = self.relu(conv6_1)

        Convolution1 = self.Convolution1(relu10)
        upsample_flow6to5 = self.upsample_flow6to5(Convolution1)
        deconv5 = self.deconv5(relu10)
        crop_upsampled_flow6to5 = crop_like(upsample_flow6to5, relu8)
        crop_deconv5 = crop_like(deconv5, relu8)
        relu11 = self.relu(crop_deconv5)

        concat2 = torch.cat((relu8, relu11, crop_upsampled_flow6to5), dim=1)
        Convolution2 = self.Convolution2(concat2)
        upsample_flow5to4 = self.upsample_flow5to4(Convolution2)
        deconv4 = self.deconv4(concat2)
        crop_upsampled_flow5to4 = crop_like(upsample_flow5to4, relu6)
        crop_deconv4 = crop_like(deconv4, relu6)
        relu12 = self.relu(crop_deconv4)

        concat3 = torch.cat((relu6, relu12, crop_upsampled_flow5to4), dim=1)
        Convolution3 = self.Convolution3(concat3)
        upsample_flow4to3 = self.upsample_flow4to3(Convolution3)
        deconv3 = self.deconv3(concat3)
        crop_upsampled_flow4to3 = crop_like(upsample_flow4to3, relu4)
        crop_deconv3 = crop_like(deconv3, relu4)
        relu13 = self.relu(crop_deconv3)

        concat4 = torch.cat((relu4, relu13, crop_upsampled_flow4to3), dim=1)
        Convolution4 = self.Convolution4(concat4)
        upsample_flow3to2 = self.upsample_flow3to2(Convolution4)
        deconv2 = self.deconv2(concat4)
        crop_upsampled_flow3to2 = crop_like(upsample_flow3to2, relu2)
        crop_deconv2 = crop_like(deconv2, relu2)
        relu14 = self.relu(crop_deconv2)

        concat5 = torch.cat((relu2, relu14, crop_upsampled_flow3to2), dim=1)
        concat5 = self.avgpool(concat5)
        Convolution5 = self.Convolution5(concat5)

        return Convolution5 * 2.5


if __name__ == '__main__':
    x = torch.rand((4, 6, 512, 512))
    f_model = FLOW_FGFA()
    y = f_model(x)
    a = 0
