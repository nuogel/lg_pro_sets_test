import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from lgdet.util.util_iou import iou_xyxy, xywh2xyxy, xyxy2xywh, iou_xywh
import numpy as np


class MULTIBOXLOSS():
    def __init__(self, cfg):
        self.cfg = cfg
        self.num_cls = self.cfg.TRAIN.CLASSES_NUM
        self.neg_iou_threshold = 0.5
        self.neg_pos_ratio = 3  # 3:1

    def Loss_Call(self, predictions, targets, kwargs):
        pre_cls, pre_loc, anc_xywh = predictions
        batchsize = pre_loc.shape[0]
        gt_images, gt_labels, infos = targets

        encode_target, labels = [], []
        for i in range(batchsize):
            gt_i = gt_labels[gt_labels[..., 0] == i]
            lab = gt_i[..., 1].long()
            box = gt_i[..., 2:]
            box_xywh = xyxy2xywh(box)
            _gt_loc_xywh, _labels = self._assign_priors(box_xywh, lab, anc_xywh[i], self.neg_iou_threshold)
            _gt_loc_xywh = self._encode_bbox(_gt_loc_xywh, anc_xywh[i])
            encode_target.append(_gt_loc_xywh)
            labels.append(_labels)
        encode_target = torch.stack(encode_target, dim=0)
        labels = torch.stack(labels, dim=0)
        num_classes = pre_cls.size(2)
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(pre_cls, dim=-1)[:, :, -1]  # -1 reprsent the background
            mask = self._hard_negative_mining(loss, labels, self.neg_pos_ratio)

        input_c = pre_cls[mask].reshape(-1, num_classes)
        gt_cls = labels[mask]
        cls_loss = F.cross_entropy(input_c, gt_cls, reduction='sum')  # F.cross_entropy函数时，程序会自动先对out进行softmax，再log，最后再计算nll_loss。
        pos_mask = labels < self.num_cls
        pre_loc = pre_loc[pos_mask, :].reshape(-1, 4)
        encode_target = encode_target[pos_mask, :].reshape(-1, 4)

        loc_loss = F.smooth_l1_loss(pre_loc, encode_target, reduction='sum')

        pos_num = pos_mask.sum()
        loc_loss = loc_loss / pos_num
        cls_loss = cls_loss / pos_num
        total_loss = loc_loss + cls_loss

        # metrics:
        with torch.no_grad():
            pre_cls = pre_cls.softmax(-1)[..., :-1]
            cls_p = (pre_cls[pos_mask].argmax(-1) == labels[pos_mask]).float().mean().item()
            pos_sc = pre_cls[pos_mask].max(-1)[0].mean().item()
            pos_t = (pre_cls[pos_mask].max(-1)[0] > self.cfg.TEST.SCORE_THRESH).float().mean().item()
            neg_sc = pre_cls[~pos_mask].max(-1)[0].mean().item()
            neg_t = (pre_cls[~pos_mask].max(-1)[0] > self.cfg.TEST.SCORE_THRESH).float().sum().item()

            metrics = {
                'pos_num': pos_num.item(),
                'pos_sc': pos_sc,
                'neg_sc': neg_sc,
                'obj>t': pos_t,
                'nob_t': neg_t,
                'cls_p': cls_p,
                'cls_loss': cls_loss.item(),
                # 'pos_loss': pos_loss.item(),
                # 'neg_loss': neg_loss.item(),
                'loc_loss': loc_loss.item(),
            }

        return {'total_loss': total_loss, 'metrics': metrics}

    def _hard_negative_mining(self, loss, labels, neg_pos_ratio):
        pos_mask = labels < self.num_cls
        num_pos = pos_mask.long().sum(dim=1, keepdim=True)
        num_neg = num_pos * neg_pos_ratio

        loss[pos_mask] = 0.
        _, indexes = loss.sort(dim=1, descending=True)
        _, orders = indexes.sort(dim=1)
        neg_mask = orders < num_neg
        return pos_mask | neg_mask

    def _assign_priors(self, gt_boxes, gt_labels, corner_form_priors, iou_threshold):
        # size: num_priors x num_targets
        ious = iou_xywh(corner_form_priors, gt_boxes, type='N2N')
        # size: num_priors
        try:
            best_target_per_prior, best_target_per_prior_index = ious.max(1)
        except:
            print('xxxx')
        # size: num_targets
        best_prior_per_target, best_prior_per_target_index = ious.max(0)

        for target_index, prior_index in enumerate(best_prior_per_target_index):
            best_target_per_prior_index[prior_index] = target_index
        # 2.0 is used to make sure every target has a prior assigned
        best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2)  # fill 2>1.
        # size: num_priors
        labels = gt_labels[best_target_per_prior_index]
        labels[best_target_per_prior < iou_threshold] = self.num_cls  # the backgournd id
        boxes = gt_boxes[best_target_per_prior_index]
        return boxes, labels

    def _encode_bbox(self, xywh_boxes, xywh_priors):
        encode_target = torch.cat([((xywh_boxes[..., :2] - xywh_priors[..., :2]) / xywh_priors[..., 2:]) / 0.1,
                                   torch.log(xywh_boxes[..., 2:] / xywh_priors[..., 2:]) / 0.2
                                   ], dim=- 1)
        return encode_target
