import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from lgdet.util.util_iou import iou_xyxy, xywh2xyxy, xyxy2xywh, iou_xywh
import numpy as np
from lgdet.loss.loss_base.focal_loss import FocalLoss_lg, FocalLoss


class RETINANETLOSS():
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.TRAIN.DEVICE
        self.num_cls = self.cfg.TRAIN.CLASSES_NUM
        self.focalloss = FocalLoss_lg()

    def Loss_Call(self, predictions, targets, kwargs):
        pre_cls, pre_loc, anc_xywh = predictions
        pre_cls = pre_cls.sigmoid()
        batchsize = pre_loc.shape[0]
        _, gt_labels, infos = targets

        gt_loc, gt_cls, pos_mask, neg_mask = [], [], [], []
        for i in range(batchsize):
            gt_i = gt_labels[gt_labels[..., 0] == i]
            lab = gt_i[..., 1].long()
            box = gt_i[..., 2:]
            box_xywh = xyxy2xywh(box)
            _gt_cls, _gt_loc_xywh, _pos_mask, _neg_mask = self._assign_priors(pre_cls[i], box_xywh, lab, anc_xywh)
            _gt_loc_xywh = self._encode_bbox(_gt_loc_xywh, anc_xywh)
            gt_loc.append(_gt_loc_xywh)
            gt_cls.append(_gt_cls)
            pos_mask.append(_pos_mask)
            neg_mask.append(_neg_mask)

        gt_loc = torch.stack(gt_loc, dim=0)
        gt_cls = torch.stack(gt_cls, dim=0)
        pos_mask = torch.stack(pos_mask, dim=0)
        neg_mask = torch.stack(neg_mask, dim=0)

        pos_num = pos_mask.sum().float()

        cls_loss, pos_loss, neg_loss = self.focalloss(pre_cls, gt_cls, pos_mask, neg_mask, reduction='sum')

        pre_loc = pre_loc[pos_mask, :].reshape(-1, 4)
        gt_loc = gt_loc[pos_mask, :].reshape(-1, 4)
        loc_loss = F.smooth_l1_loss(pre_loc, gt_loc, reduction='sum')

        pos_loss /= pos_num
        neg_loss /= pos_num
        loc_loss /= pos_num
        total_loss = loc_loss + pos_loss + neg_loss

        # metrics:
        cls_p = (pre_cls[pos_mask].argmax(-1) == gt_cls[pos_mask].argmax(-1)).float().mean().item()
        pos_sc = pre_cls[pos_mask].max(-1)[0].mean().item()
        pos_t = (pre_cls[pos_mask].max(-1)[0] > self.cfg.TEST.SCORE_THRESH).float().mean().item()
        neg_sc = pre_cls[neg_mask].max(-1)[0].mean().item()
        neg_t = (pre_cls[neg_mask].max(-1)[0] > self.cfg.TEST.SCORE_THRESH).float().sum().item()

        metrics = {
            'pos_num': pos_num.item(),
            'pos_sc': pos_sc,
            'neg_sc': neg_sc,
            'obj>t': pos_t,
            'nob>t': neg_t,
            'cls_p': cls_p,

            'pos_loss': pos_loss.item(),
            'neg_loss': neg_loss.item(),
            'loc_loss': loc_loss.item(),
        }
        return {'total_loss': total_loss, 'metrics': metrics}

    def _assign_priors(self, pre_cls, gt_boxes, gt_labels, corner_form_priors):
        ious = iou_xywh(corner_form_priors, gt_boxes, type='N2N')
        best_target_per_prior, best_target_per_prior_index = ious.max(1)

        best_prior_per_target, best_prior_per_target_index = ious.max(0)
        # #
        for target_index, prior_index in enumerate(best_prior_per_target_index):
            best_target_per_prior_index[prior_index] = target_index
        # 2.0 is used to make sure every target has a prior assigned
        best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2)  # fill 2>1.
        # size: num_priors

        gt_cls = torch.zeros_like(pre_cls)
        pos_mask = best_target_per_prior > 0.5
        neg_mask = best_target_per_prior < 0.4

        # gt_cls[neg_mask, -1] = 1  # background set to one hot [0, ..., 1]
        assigned_annotations = gt_labels[best_target_per_prior_index]

        # gt_cls[pos_mask, :] = 0
        gt_cls[pos_mask, assigned_annotations[pos_mask].long()] = 1
        gt_loc = gt_boxes[best_target_per_prior_index]
        return gt_cls, gt_loc, pos_mask, neg_mask

    def _encode_bbox(self, xywh_boxes, xywh_priors):
        encode_target = torch.cat([(xywh_boxes[..., :2] - xywh_priors[..., :2]) / xywh_priors[..., 2:] / 0.1,
                                   torch.log(xywh_boxes[..., 2:] / xywh_priors[..., 2:]) / 0.2
                                   ], dim=- 1)
        return encode_target
