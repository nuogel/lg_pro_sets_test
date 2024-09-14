import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from lgdet.util.util_anchor_maker_lg import PriorBox
import numpy as np

from ..registry import MODELS


@MODELS.registry()
class REFINEDET(nn.Module):

    def __init__(self, cfg):
        super(REFINEDET, self).__init__()
        self.cfg = cfg
        size = self.cfg.TRAIN.IMG_SIZE[0]
        self.num_classes = cfg.TRAIN.CLASSES_NUM

        phase = 'train'
        self.init(phase, size, self.num_classes)
        self.phase = phase
        self.anchors = PriorBox(image_shape=self.cfg.TRAIN.IMG_SIZE, pyramid_levels=[3, 4, 5, 6], scales=np.array([1]))
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(self.base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.conv4_3_L2Norm = L2Norm(512, 10)
        self.conv5_3_L2Norm = L2Norm(512, 8)
        self.extras = nn.ModuleList(self.extras_)

        self.arm_loc = nn.ModuleList(self.ARM[0])
        self.arm_conf = nn.ModuleList(self.ARM[1])
        self.odm_loc = nn.ModuleList(self.ODM[0])
        self.odm_conf = nn.ModuleList(self.ODM[1])
        # self.tcb = nn.ModuleList(TCB)
        self.tcb0 = nn.ModuleList(self.TCB[0])
        self.tcb1 = nn.ModuleList(self.TCB[1])
        self.tcb2 = nn.ModuleList(self.TCB[2])

    def init(self, phase, size, num_classes):
        if phase != "test" and phase != "train":
            print("ERROR: Phase: " + phase + " not recognized")
            return
        if size != 320 and size != 512:
            print("ERROR: You specified size " + repr(size) + ". However, " +
                  "currently only RefineDet320 and RefineDet512 is supported!")
            return
        self.base = vgg(base[str(size)], 3)
        self.extras_ = add_extras(extras[str(size)], size, 1024)
        self.ARM = self.arm_multibox(self.base, self.extras_, mbox[str(size)])
        self.ODM = self.odm_multibox(self.base, self.extras_, mbox[str(size)], num_classes)
        self.TCB = self.add_tcb(tcb[str(size)])

    def forward(self, **args):

        x = args['input_x']

        sources = list()
        tcb_source = list()
        arm_loc = list()
        arm_conf = list()
        odm_loc = list()
        odm_conf = list()

        # apply vgg up to conv4_3 relu and conv5_3 relu
        for k in range(30):
            x = self.vgg[k](x)
            if 22 == k:
                s = self.conv4_3_L2Norm(x)
                sources.append(s)
            elif 29 == k:
                s = self.conv5_3_L2Norm(x)
                sources.append(s)

        # apply vgg up to fc7
        for k in range(30, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply ARM and ODM to source layers
        for (x, l, c) in zip(sources, self.arm_loc, self.arm_conf):
            arm_loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            arm_conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        arm_loc = torch.cat([o.view(o.size(0), -1) for o in arm_loc], 1)
        arm_conf = torch.cat([o.view(o.size(0), -1) for o in arm_conf], 1)
        # print([x.size() for x in sources])
        # calculate TCB features
        # print([x.size() for x in sources])
        p = None
        for k, v in enumerate(sources[::-1]):
            s = v
            for i in range(3):
                s = self.tcb0[(3 - k) * 3 + i](s)
                # print(s.size())
            if k != 0:
                u = p
                u = self.tcb1[3 - k](u)
                s += u
            for i in range(3):
                s = self.tcb2[(3 - k) * 3 + i](s)
            p = s
            tcb_source.append(s)
        # print([x.size() for x in tcb_source])
        tcb_source.reverse()

        # apply ODM to source layers
        for (x, l, c) in zip(tcb_source, self.odm_loc, self.odm_conf):
            odm_loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            odm_conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        odm_loc = torch.cat([o.view(o.size(0), -1) for o in odm_loc], 1)
        odm_conf = torch.cat([o.view(o.size(0), -1) for o in odm_conf], 1)
        # print(arm_loc.size(), arm_conf.size(), odm_loc.size(), odm_conf.size())

        output = (arm_loc.view(arm_loc.size(0), -1, 4),
                  arm_conf.view(arm_conf.size(0), -1, 2),
                  odm_loc.view(odm_loc.size(0), -1, 4),
                  odm_conf.view(odm_conf.size(0), -1, self.num_classes),
                  # self.priorbox.forward().type(type(x)),  # raw code anchor
                  self.anchors.forward().type(type(x))  # lg anchor
                  )

        return output

    def arm_multibox(self, vgg, extra_layers, cfg):
        arm_loc_layers = []
        arm_conf_layers = []
        vgg_source = [21, 28, -2]
        for k, v in enumerate(vgg_source):
            arm_loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                         cfg[k] * 4, kernel_size=3, padding=1)]
            arm_conf_layers += [nn.Conv2d(vgg[v].out_channels,
                                          cfg[k] * 2, kernel_size=3, padding=1)]
        for k, v in enumerate(extra_layers[1::2], 3):
            arm_loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                         * 4, kernel_size=3, padding=1)]
            arm_conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                          * 2, kernel_size=3, padding=1)]
        return (arm_loc_layers, arm_conf_layers)

    def odm_multibox(self, vgg, extra_layers, cfg, num_classes):
        odm_loc_layers = []
        odm_conf_layers = []
        vgg_source = [21, 28, -2]
        for k, v in enumerate(vgg_source):
            odm_loc_layers += [nn.Conv2d(256, cfg[k] * 4, kernel_size=3, padding=1)]
            odm_conf_layers += [nn.Conv2d(256, cfg[k] * num_classes, kernel_size=3, padding=1)]
        for k, v in enumerate(extra_layers[1::2], 3):
            odm_loc_layers += [nn.Conv2d(256, cfg[k] * 4, kernel_size=3, padding=1)]
            odm_conf_layers += [nn.Conv2d(256, cfg[k] * num_classes, kernel_size=3, padding=1)]
        return (odm_loc_layers, odm_conf_layers)

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                            map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

    def add_tcb(self, cfg):
        feature_scale_layers = []
        feature_upsample_layers = []
        feature_pred_layers = []
        for k, v in enumerate(cfg):
            feature_scale_layers += [nn.Conv2d(cfg[k], 256, 3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(256, 256, 3, padding=1)
                                     ]
            feature_pred_layers += [nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, 3, padding=1),
                                    nn.ReLU(inplace=True)
                                    ]
            if k != len(cfg) - 1:
                feature_upsample_layers += [nn.ConvTranspose2d(256, 256, 2, 2)]
        return (feature_scale_layers, feature_upsample_layers, feature_pred_layers)


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
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
    pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=3, dilation=3)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, size, i, batch_norm=False):
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
    return layers


base = {
    '320': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}
extras = {
    '320': [256, 'S', 512],
    '512': [256, 'S', 512],
}
mbox = {
    '320': [3, 3, 3, 3],  # number of boxes per feature map location
    '512': [3, 3, 3, 3],  # number of boxes per feature map location
}

tcb = {
    '320': [512, 512, 1024, 512],
    '512': [512, 512, 1024, 512],
}


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        # x /= norm
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

# class PriorBox(object):
#     """Compute priorbox coordinates in center-offset form for each source
#     feature map.
#     """
#
#     def __init__(self, cfg):
#         super(PriorBox, self).__init__()
#         self.image_size = cfg['min_dim']
#         # number of priors for feature map location (either 4 or 6)
#         self.num_priors = len(cfg['aspect_ratios'])
#         self.variance = cfg['variance'] or [0.1]
#         self.feature_maps = cfg['feature_maps']
#         self.min_sizes = cfg['min_sizes']
#         self.max_sizes = cfg['max_sizes']
#         self.steps = cfg['steps']
#         self.aspect_ratios = cfg['aspect_ratios']
#         self.clip = cfg['clip']
#         self.version = cfg['name']
#         for v in self.variance:
#             if v <= 0:
#                 raise ValueError('Variances must be greater than 0')
#
#     def forward(self):
#         mean = []
#         for k, f in enumerate(self.feature_maps):
#             for i, j in product(range(f), repeat=2):
#                 f_k = self.image_size / self.steps[k]
#                 # unit center x,y
#                 cx = (j + 0.5) / f_k
#                 cy = (i + 0.5) / f_k
#
#                 # aspect_ratio: 1
#                 # rel size: min_size
#                 s_k = self.min_sizes[k] / self.image_size
#                 mean += [cx, cy, s_k, s_k]
#
#                 # aspect_ratio: 1
#                 # rel size: sqrt(s_k * s_(k+1))
#                 if self.max_sizes:
#                     s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
#                     mean += [cx, cy, s_k_prime, s_k_prime]
#
#                 # rest of aspect ratios
#                 for ar in self.aspect_ratios[k]:
#                     mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
#                     mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]
#         # back to torch land
#         output = torch.Tensor(mean).view(-1, 4)
#         if self.clip:
#             output.clamp_(max=1, min=0)
#         return output
# voc_refinedet = {
#     '320': {
#         'num_classes': 21,
#         'lr_steps': (80000, 100000, 120000),
#         'max_iter': 120000,
#         'feature_maps': [40, 20, 10, 5],
#         'min_dim': 320,
#         'steps': [8, 16, 32, 64],
#         'min_sizes': [32, 64, 128, 256],
#         'max_sizes': [],
#         'aspect_ratios': [[2], [2], [2], [2]],
#         'variance': [0.1, 0.2],
#         'clip': True,
#         'name': 'RefineDet_VOC_320',
#     },
#     '512': {
#         'num_classes': 21,
#         'lr_steps': (80000, 100000, 120000),
#         'max_iter': 120000,
#         'feature_maps': [64, 32, 16, 8],
#         'min_dim': 512,
#         'steps': [8, 16, 32, 64],
#         'min_sizes': [32, 64, 128, 256],
#         'max_sizes': [],
#         'aspect_ratios': [[2], [2], [2], [2]],
#         'variance': [0.1, 0.2],
#         'clip': True,
#         'name': 'RefineDet_VOC_320',
#     }
# }
