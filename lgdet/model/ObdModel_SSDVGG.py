import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from lgdet.util.util_anchor_maker_lg import PriorBox

vgg_base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}
_extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256],
}
mbox = {
    '300': [6, 6, 6, 6, 6, 6],  # number of boxes per feature map location
    '512': [6, 6, 6, 6, 6, 6, 6],
}

from ..registry import MODELS


@MODELS.registry()
class SSDVGG(nn.Module):
    """Single Shot Multibox Architecture  SSDVGG
    """

    def __init__(self, cfg):
        super(SSDVGG, self).__init__()
        self.cfg = cfg
        self.num_classes = cfg.TRAIN.CLASSES_NUM + 1
        # TODO: implement __call__ in PriorBox
        self.size = 512
        base, extras, head = self.multibox(vgg(vgg_base[str(self.size)], 3),
                                           self.add_extras(_extras[str(self.size)], 1024, size=self.size),
                                           mbox[str(self.size)],
                                           self.num_classes)

        # SSD network
        self.base = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.extras = nn.ModuleList(extras)
        # self.L2Norm = L2Norm(512, 20)
        self.L2Norm = nn.BatchNorm2d(512)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        self.priorbox = PriorBox(image_shape=self.cfg.TRAIN.IMG_SIZE, pyramid_levels=[3, 4, 5, 6, 7, 8, 9], scales=np.array([1, 1.25]))
        self.anchors_xywh = self.priorbox.forward()
        self.softmax = nn.Softmax()

    def forward(self, **args):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        x = args['input_x']

        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.base[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.base)):
            x = self.base[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        loc_layers = [o.view(o.size(0), -1) for o in loc]
        conf_layers = [o.view(o.size(0), -1) for o in conf]
        loc = torch.cat(loc_layers, 1)
        conf = torch.cat(conf_layers, 1)

        output = (
            conf.view(conf.size(0), -1, self.num_classes),
            loc.view(loc.size(0), -1, 4),
            self.anchors_xywh.to(loc.device)
        )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

    def add_extras(self, cfg, i, batch_norm=False, size=300):
        # Extra layers added to VGG for feature scaling
        layers = []
        in_channels = i
        flag = False
        for k, v in enumerate(cfg):
            if in_channels != 'S':
                if v == 'S':
                    layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                         kernel_size=(1, 3)[flag], stride=2, padding=1)]
                else:
                    layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
                flag = not flag
            in_channels = v
        if size == 512:
            layers.append(nn.Conv2d(in_channels, 128, kernel_size=1, stride=1))
            layers.append(nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1))
        return layers

    def multibox(self, vgg, extra_layers, cfg, num_classes):
        loc_layers = []
        conf_layers = []
        vgg_source = [24, -2]
        for k, v in enumerate(vgg_source):
            loc_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]

        for k, v in enumerate(extra_layers[1::2], 2):
            loc_layers += [nn.Conv2d(v.out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(v.out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]

        return vgg, extra_layers, (loc_layers, conf_layers)


# def build_net(size=300, num_classes=21):
#     if size != 300 and size != 512:
#         print("Error: Sorry only SSD300 and SSD512 is supported currently!")
#         return
#
#     return SSD(*multibox(vgg(vgg_base[str(size)], 3), add_extras(extras[str(size)], 1024, size=size), mbox[str(size)], num_classes), num_classes=num_classes, size=size)
#

def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x /= norm
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out
