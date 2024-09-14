import torch
from lgdet.util.util_refinedet import decode, center_size, log_sum_exp, refine_match
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Detect_RefineDet():
    def __init__(self, cfg):
        self.num_classes = cfg.TRAIN.CLASSES_NUM
        self.top_k = 1000
        self.keep_top_k = 500
        self.objectness_thre = 0.01
        self.variance = [0.1, 0.2]

    def forward(self, predicts):
        arm_loc_data, arm_conf_data, odm_loc_data, odm_conf_data, prior_data = predicts

        arm_conf_data = torch.softmax(arm_conf_data.view(arm_conf_data.size(0), -1, 2), -1)
        odm_conf_data = torch.softmax(odm_conf_data.view(odm_conf_data.size(0), -1, self.num_classes), -1)
        loc_data = odm_loc_data
        conf_data = odm_conf_data

        arm_object_conf = arm_conf_data.data[:, :, 1:]
        no_object_index = arm_object_conf <= self.objectness_thre
        conf_data[no_object_index.expand_as(conf_data)] = 0

        num = loc_data.size(0)  # batch size

        # Decode predictions into bboxes.
        scores_out = conf_data
        boxes_out = []
        for i in range(num):
            default = decode(arm_loc_data[i], prior_data.to(arm_loc_data[i].device), self.variance)
            default = center_size(default)
            decoded_boxes = decode(loc_data[i], default, self.variance)
            # For each class, perform nms
            boxes_out.append(decoded_boxes)
        boxes_out = torch.stack(boxes_out, 0)
        output = (scores_out, boxes_out)
        return output


class RefineDetMultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, cfg, num_classes, use_ARM=False):
        super(RefineDetMultiBoxLoss, self).__init__()
        self.cfg = cfg
        self.use_gpu = True
        self.num_classes = num_classes
        self.threshold = 0.5
        self.background_label = 0
        self.encode_target = False
        self.use_prior_for_matching = True
        self.do_neg_mining = True
        self.negpos_ratio = 3
        self.neg_overlap = 0.5
        self.variance = [0.1, 0.2]
        self.theta = 0.01
        self.use_ARM = use_ARM

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        # TODO: make a combination with SSD.
        arm_loc_data, arm_conf_data, odm_loc_data, odm_conf_data, priors = predictions
        # print(arm_loc_data.size(), arm_conf_data.size(),
        #      odm_loc_data.size(), odm_conf_data.size(), priors.size())
        # input()
        if self.use_ARM:
            loc_data, conf_data = odm_loc_data, odm_conf_data
        else:
            loc_data, conf_data = arm_loc_data, arm_conf_data
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes
        # print(loc_data.size(), conf_data.size(), priors.size())

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            target = torch.Tensor(np.array(targets[idx]))
            truths = target[:, 1:].to(arm_loc_data.device)
            labels = target[:, 0]
            if num_classes == 2:
                labels = labels >= 0
            defaults = priors.to(truths.device)
            if self.use_ARM:
                loc_t, conf_t = refine_match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx, arm_loc_data[idx].data)
            else:
                loc_t, conf_t = refine_match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)
        if self.use_gpu:
            loc_t = loc_t.to(self.cfg.TRAIN.DEVICE)
            conf_t = conf_t.to(self.cfg.TRAIN.DEVICE)
        # wrap targets
        # loc_t = Variable(loc_t, requires_grad=False)
        # conf_t = Variable(conf_t, requires_grad=False)
        loc_t.requires_grad = False
        conf_t.requires_grad = False
        # print(loc_t.size(), conf_t.size())

        if self.use_ARM:
            P = F.softmax(arm_conf_data, 2)
            arm_conf_tmp = P[:, :, 1]
            object_score_index = arm_conf_tmp <= self.theta
            pos = conf_t > 0
            pos[object_score_index.data] = 0
        else:
            pos = conf_t > 0
        # print(pos.size())
        # num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)

        # batch_conf = batch_conf.clamp(-10., 10.)

        a = log_sum_exp(batch_conf)
        b = batch_conf.gather(1, conf_t.view(-1, 1))
        loss_c = a - b
        # print(loss_c.size())

        # Hard Negative Mining
        loss_c[pos.view(-1, 1)] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        # print(num_pos.size(), num_neg.size(), neg.size())

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos + neg).gt(0)]
        # print(pos_idx.size(), neg_idx.size(), conf_p.size(), targets_weighted.size())
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')  # conf_p不用softmax()等一系列处理。。。见博客

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum().float()
        # N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N
        # print(N, loss_l, loss_c)
        return 5*loss_l, loss_c


class REFINEDETLOSS:
    def __init__(self, cfg):
        self.cls_num = cfg.TRAIN.CLASSES_NUM
        self.arm_criterion = RefineDetMultiBoxLoss(cfg, 2, use_ARM=False)
        self.odm_criterion = RefineDetMultiBoxLoss(cfg, self.cls_num, use_ARM=True)

    def Loss_Call(self, predict, dataset, losstype):
        targets = dataset[1]
        arm_loss_l, arm_loss_c = self.arm_criterion(predict, targets)
        odm_loss_l, odm_loss_c = self.odm_criterion(predict, targets)
        return arm_loss_l, arm_loss_c, odm_loss_l, odm_loss_c
