import torch.nn as nn
import torch
import torch.nn.functional as F
from lgdet.util.util_wap_FLOW import warp
from lgdet.model.FlowModel_FLOW_FGFA import FLOW_FGFA
from lgdet.model.aid_models.Model_YOLOV2_backbone import YOLOV2_backbone


class EmbedNet(nn.Module):
    def __init__(self):
        super(EmbedNet, self).__init__()
        self.em_conv1 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1)  # in ch  TODO
        self.em_conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)  # in ch
        self.em_conv3 = nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=1, stride=1)  # in ch

    def forward(self, x):
        x = F.relu(self.em_conv1(x))
        x = F.relu(self.em_conv2(x))
        x = self.em_conv3(x)
        return x


class FGFA(nn.Module):
    def __init__(self, cfg=None):
        super(FGFA, self).__init__()
        self.backbone = YOLOV2_backbone()
        self.embednet = EmbedNet()
        self.flownet = FLOW_FGFA()

        for param in self.flownet.parameters():
            param.requires_grad = False

        self.loaded = 0

    def compute_weight(self, embed_flow, embed_conv_feat):
        def l2normalization(tensor):
            norm = torch.norm(tensor, dim=1, keepdim=True) + 1e-10
            return tensor / norm

        embed_flow_norm = l2normalization(embed_flow)
        embed_conv_norm = l2normalization(embed_conv_feat)
        weight = torch.sum(embed_flow_norm * embed_conv_norm, dim=1, keepdim=True)
        return weight

    def forward(self, ref_images, guide_images):
        """
        ref_images: tensor([1, c, h, w])
        guided_images: list[tensor([c, h, w])]

        return: flow-guided aggregation feature map of which shape is tensor([b, c, h, w])
        """
        num_refs = len(guide_images)

        concat_imgs = torch.cat([ref_images, guide_images], dim=0)
        concat_feats = self.backbone(concat_imgs)

        # img_cur, imgs_ref = torch.split(concat_imgs, (1, num_refs), dim=0)
        img_cur_copies = ref_images.repeat(num_refs, 1, 1, 1)
        concat_imgs_pair = torch.cat([img_cur_copies, guide_images], dim=1)
        if self.loaded == 0:
            self.checkpoint = torch.load('tmp/checkpoint/FGFA.pkl')
            self.flownet.load_state_dict(self.checkpoint)
            self.loaded = 1

        flow = self.flownet(input_x=concat_imgs_pair)

        feats_cur, feats_refs = torch.split(concat_feats, (1, num_refs), dim=0)
        warped_feats_refs = warp(feats_refs, flow)

        concat_feats = torch.cat([feats_cur, warped_feats_refs], dim=0)
        concat_embed_feats = self.embednet(concat_feats)
        embed_cur, embed_refs = torch.split(concat_embed_feats, (1, num_refs), dim=0)

        unnormalized_weights = self.compute_weight(embed_refs, embed_cur)
        weights = F.softmax(unnormalized_weights, dim=0)

        feats = torch.sum(weights * warped_feats_refs, dim=0, keepdim=True)
        # print(feats.shape)
        return feats  # feats_cur #


from ..registry import MODELS


@MODELS.registry()
class YOLOV2_FGFA(nn.Module):
    def __init__(self, cfg=None):
        super(YOLOV2_FGFA, self).__init__()
        self.cfg = cfg
        self.head = FGFA(self.cfg)
        if cfg is None:
            self.anc_num = 6
            self.cls_num = 4
        else:
            self.anc_num = cfg.TRAIN.FMAP_ANCHOR_NUM
            self.cls_num = cfg.TRAIN.CLASSES_NUM
        self.tail = nn.Conv2d(1024, self.anc_num * (5 + self.cls_num), 1, 1, 0, bias=False)

    def forward(self, **args):
        data = args['input_x']
        x = data[0].unsqueeze(0)
        ref = data[1:]
        x = self.head(x, ref)
        x = self.tail(x)
        x = x.permute([0, 2, 3, 1])
        # print(x.shape)
        return [x, ]


if __name__ == '__main__':
    yolofgfa = YOLOV2_FGFA().cuda()
    x = torch.rand(10, 3, 512, 512).cuda()
    out = yolofgfa(input_x=x)
    a = 0
