# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions from official
"""

import torch
import torch.nn as nn
import numpy as np
from lgdet.util.util_iou import _iou_wh, bbox_GDCiou, xywh2xyxy, iou_xyxy, wh_iou
from lgdet.postprocess.parse_factory import ParsePredict

from lgdet.loss.loss_base.focal_loss import FocalLoss_lg
from lgdet.loss.loss_base.ghm_loss import GHMC
from lgdet.loss.loss_base.smothBCE import smooth_BCE


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class YoloLoss:
    # pylint: disable=too-few-public-methods
    """Calculate loss."""

    def __init__(self, cfg):
        """Init."""
        super(YoloLoss, self).__init__()
        self.cfg = cfg
        self.device = self.cfg.TRAIN.DEVICE
        self.anchors = torch.Tensor(cfg.TRAIN.ANCHORS[::-1]).reshape((3, 3, 2))
        self.anc_num = cfg.TRAIN.FMAP_ANCHOR_NUM
        self.na = self.anc_num
        self.cls_num = cfg.TRAIN.CLASSES_NUM
        self.one_test = cfg.TEST.ONE_TEST
        self.base = 6 if self.cfg.TRAIN.IOU_AWARE else 5

        self.reduction = 'mean'
        self.parsepredict = ParsePredict(cfg)
        self.multiply_area_scale = 0  # whether multiply loss to area_scale.

        self.alpha = 0.25
        self.gamma = 0
        self.sort_obj_iou = False
        self.balance = [4.0, 1.0, 0.4]
        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(self.device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(self.device))
        # Focal loss
        g = self.gamma  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
        self.BCEcls, self.BCEobj, self.gr, = BCEcls, BCEobj, 1.0,
        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=0.0)  # positive, negative BCE targets

    def _build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(3):
            B, C, H, W = p[i].shape
            anchors_raw = self.anchors[i]
            anchors = torch.Tensor([(a_w / self.cfg.TRAIN.IMG_SIZE[1] * W, a_h / self.cfg.TRAIN.IMG_SIZE[0] * H) for a_w, a_h in anchors_raw]).to(self.device)

            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                self.anchor_t = 4.0
                j = torch.max(r, 1. / r).max(2)[0] < self.anchor_t  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch

    # def __call__(self, p, targets):  # predictions, targets, model
    def Loss_Call(self, p, dataset, kwargs):
        images, labels, datainfos = dataset
        targets = labels.clone()
        targets[..., 2:4] = (labels[..., 2:4] + labels[..., 4:6]) / 2
        targets[..., 4:6] = (labels[..., 4:6] - labels[..., 2:4])

        lcls, lbox, lobj,noobj_t = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device), torch.zeros(1, device=self.device),torch.zeros(1, device=self.device)
        meanious = []
        pos_score = []
        cls_score = []
        tcls, tbox, indices, anchors = self._build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            B, C, H, W = pi.shape
            pi = pi.view(B, self.anc_num, self.cls_num + self.base, H, W)
            permiute_type = (0, 1, 3, 4, 2)
            pi = pi.permute(permiute_type).contiguous()
            # pred_conf, pred_cls, pred_xy, pred_wh, pred_iou, pre_relative_box = self.parsepredict.parser._parse_yolo_predict_fmap(pi, i)
            # pre_obj, pre_cls, pre_loc_xy, pre_loc_wh = pred_conf.value, pred_cls.value, pred_xy.value, pred_wh.value

            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pre_loc_xy = ps[:, 1:3].sigmoid() * 2. - 0.5
                pre_loc_wh = (ps[:, 3:5].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pre_loc_xy, pre_loc_wh), 1)  # predicted box
                iou = bbox_GDCiou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss
                meanious.append(iou)
                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                if self.cls_num > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, self.base:], self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, self.base:], t)  # BCE
                    cls_score.append(((ps[:, self.base:].max(-1)[1] == tcls[i]).float()))

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]
            obji = self.BCEobj(pi[..., 0], tobj)
            noobj_t += (((pi[..., 0][tobj == 0]).sigmoid()) > 0.4).sum()
            lobj += obji * self.balance[i]  # obj loss
            pos_score.append((pi[..., 0][b, a, gj, gi]).sigmoid())

        meaniou = torch.cat(meanious, -1).mean()
        cls_score = torch.cat(cls_score, -1).mean()
        pos_score = torch.cat(pos_score, -1).mean()

        bs = tobj.shape[0]  # batch size
        lbox *= 1  # 0.05
        lcls *= 5 * self.cls_num / 80  # 0.5 * nc / 80 * 3 / nl
        lobj *= 10  # 1
        total_loss = (lbox + lobj + lcls) / bs
        metrics = {'box_loss': lbox.item(),
                   'obj_loss': lobj.item(),
                   'cls_loss': lcls.item(),
                   'mean_iou': meaniou.item(),
                   'cls_p': cls_score.item(),
                   'pos_score': pos_score.item(),
                   'noobj_t':noobj_t.item()/B}
        return {'total_loss': total_loss, 'metrics': metrics}
