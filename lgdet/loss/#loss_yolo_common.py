"""Loss calculation based on yolo."""
import torch
import numpy as np
from lgdet.util.util_iou import _iou_wh, bbox_GDCiou, xywh2xyxy, iou_xyxy, wh_iou, xyxy2xywh
from lgdet.postprocess.parse_factory import ParsePredict

from lgdet.loss.loss_base.focal_loss import FocalLoss, FocalLoss_lg
from lgdet.loss.loss_base.ghm_loss import GHMC
from lgdet.loss.loss_base.smothBCE import smooth_BCE

'''
with the new yolo loss, in 56 images, loss is 0.18 and map is 0.2.and the test show wrong bboxes.
with the new yolo loss, in 8 images, loss is 0.015 and map is 0.99.and the test show terrible bboxes.

'''


class YoloLoss:
    # pylint: disable=too-few-public-methods
    """Calculate loss."""

    def __init__(self, cfg):
        """Init."""
        #
        self.cfg = cfg
        self.device = self.cfg.TRAIN.DEVICE
        self.anchors = torch.Tensor(cfg.TRAIN.ANCHORS)
        self.anc_num = cfg.TRAIN.FMAP_ANCHOR_NUM
        self.cls_num = cfg.TRAIN.CLASSES_NUM
        self.one_test = cfg.TEST.ONE_TEST

        self.reduction = 'mean'
        self.mseloss = torch.nn.MSELoss(reduction=self.reduction)
        self.bceloss = torch.nn.BCELoss(reduction=self.reduction)

        self.parsepredict = ParsePredict(cfg)
        self.multiply_area_scale = 0  # whether multiply loss to area_scale.

        self.alpha = 0.25
        self.gamma = 2
        self.Focalloss = FocalLoss(loss_weight=1.0, pos_weight=1.0, gamma=1.5, alpha=0.25, reduction='mean').to(self.device)
        self.Focalloss_lg = FocalLoss_lg(alpha=self.alpha, gamma=self.gamma, ).to(self.device)
        self.ghm = GHMC(use_sigmoid=True)

    def build_targets(self, pre_obj, labels, f_id):
        # time_1 = time.time()
        targets = labels
        B, C, H, W = pre_obj.shape

        mask = np.arange(self.anc_num) + self.anc_num * f_id  # be care of the relationship between anchor size and the feature map size.
        anchors_raw = self.anchors[mask]
        anchors = torch.Tensor([(a_w / self.cfg.TRAIN.IMG_SIZE[1] * W, a_h / self.cfg.TRAIN.IMG_SIZE[0] * H) for a_w, a_h in anchors_raw]).to(self.device)

        num_target = targets.shape[0]
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        na = anchors.shape[0]  # number of anchors

        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, num_target)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        gain[2:6] = torch.tensor([W, H, W, H])  # xyxy gain
        t = targets * gain
        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        # Match targets to anchors
        if num_target:
            # Matches
            r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
            j = torch.max(r, 1. / r).max(2)[0] < 4.  # compare
            # j = wh_iou(anchors, t[:, 4:6]) > 0.2  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
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

        a = t[:, 6].long()  # anchor indices
        indices = [b, a, gj, gi]  # image, anchor, grid indices

        tbox = torch.cat((gxy - gij, gwh), 1)  # box
        anch = anchors[a]  # anchors
        tcls = c  # class
        return tcls, tbox, indices, anch

    def _loss_cal_one_Fmap(self, f_map, f_id, labels, kwargs):
        """Calculate the loss."""
        # init loss.
        loc_losstype, obj_losstype = kwargs['losstype']
        balance = [4.0, 1.0, 0.4][f_id]
        metrics = {}
        lcls, lbox, lobj, loss_iou_aware, iou = [torch.FloatTensor([0]).to(self.device) for _ in range(5)]
        pre_obj, pre_cls, pre_loc_xy, pre_loc_wh, pred_iou, pre_relative_box = self.parsepredict.parser._parse_yolo_predict_fmap(f_map, f_id)
        with torch.no_grad():
            tcls, tbox, indices, anchors = self.build_targets(pre_obj, labels, f_id)  # targets
            B, C, H, W = pre_obj.shape
            tobj = torch.zeros_like(pre_obj)  # target obj
            num_target = anchors.shape[0]  # number of targets
            loss_ratio = {'box': 0.05, 'cls': 0.125, 'obj': 1 * balance, 'noobj': 1 * balance}

        if num_target:
            pre_cls, pre_xy, pre_wh = [i[indices] for i in [pre_cls, pre_loc_xy, pre_loc_wh]]
            lcls = self.loss_cls(pre_cls, num_target, tcls, metrics)

            # 1) boxes loss.
            if self.multiply_area_scale:
                area = tbox[..., 2] * tbox[..., 3] / (H * W)
                area_scale = (1 - torch.sqrt(area) * 2).clamp(0.2, 1)  # 2 is scale parameter
            else:
                area_scale = 1.

            if loc_losstype == 'mse':
                # mse:
                tobj[indices] = 1.0
                twh = torch.log(tbox[..., 2:] / anchors + 1e-10)  # torch.log(gw / anchors[best_n][:, 0] + 1e-16)
                lxy = self.mseloss(pre_xy, tbox[..., :2])
                if self.multiply_area_scale: area_scale = area_scale.unsqueeze(-1).expand_as(pre_wh)
                lwh = self.mseloss(pre_wh * area_scale, twh * area_scale)
                lbox = (lxy + lwh)

            elif loc_losstype == 'iouloss':
                pre_wh = pre_wh * anchors
                pbox = torch.cat((pre_xy, pre_wh), 1)  # predicted box
                iou = bbox_GDCiou(pbox.t(), tbox, x1y1x2y2=False, CIoU=True)  # giou(prediction, target)
                lbox = (1.0 - iou) * area_scale
                if self.reduction == 'mean':
                    lbox = lbox.mean()  # giou loss
                else:
                    lbox = lbox.sum() / num_target  # giou loss
                # # Objectness
                global_step = kwargs['global_step']
                if self.one_test:
                    gr = 1
                else:
                    len_batch = kwargs['len_batch']
                    xi = [0, max(3 * len_batch, 500)]
                    gr = np.interp(global_step, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                tobj[indices] = (1.0 - gr) + gr * iou.detach().clamp(min=0.1).type(tobj.dtype)  # giou ratio

                if self.cfg.TRAIN.IOU_AWARE:
                    iou = torch.clamp(iou, 0, 1)
                    pred_iou = torch.clamp(pred_iou[indices], 0, 1)
                    loss_iou_aware = self.bceloss(pred_iou, iou.detach())

        obj_mask = tobj > 0.
        noobj_mask = ~obj_mask
        # noobj_mask[indices_ignore] = False
        label_weight_mask = (obj_mask | noobj_mask)
        obj_num = obj_mask.sum()
        if obj_losstype == 'focalloss':
            pre_obj = pre_obj.sigmoid()
            _loss, obj_loss, noobj_loss = self.Focalloss_lg(pre_obj, tobj, obj_mask, noobj_mask, split_loss=True)
        if obj_losstype == 'ghm':
            obj_loss = self.ghm(pre_obj, tobj, label_weight_mask)
            noobj_loss = torch.FloatTensor([0]).to(self.device)
        elif obj_losstype == 'bce':
            if obj_num:
                obj_loss = self.bceloss(pre_obj[obj_mask], tobj[obj_mask])  # obj loss
            else:
                obj_loss = torch.FloatTensor([0]).to(self.device)
            noobj_loss = self.bceloss(pre_obj[noobj_mask], tobj[noobj_mask])  # obj loss
        elif obj_losstype == 'mse':
            if obj_num:
                obj_loss = self.mseloss(pre_obj[obj_mask], tobj[obj_mask])  # obj loss
            else:
                obj_loss = torch.FloatTensor([0]).to(self.device)
            noobj_loss = self.mseloss(pre_obj[noobj_mask], tobj[noobj_mask])  # obj loss

        obj_loss = loss_ratio['obj'] * obj_loss
        noobj_loss = loss_ratio['noobj'] * noobj_loss
        lcls = loss_ratio['cls'] * lcls
        lbox = loss_ratio['box'] * lbox

        total_loss = obj_loss + noobj_loss + lcls + lbox + loss_iou_aware

        if self.reduction == 'sum': total_loss = total_loss / B
        if torch.isnan(total_loss) or total_loss.item() == float("inf") or total_loss.item() == -float("inf"):
            print('nan')

        # metrics
        pre_obj = pre_obj.sigmoid()
        obj_sc = pre_obj[obj_mask].mean().item() if obj_num > 0. else 0.
        noobj_sc = pre_obj[noobj_mask].mean().item()
        obj_percent = ((pre_obj[obj_mask] > self.cfg.TEST.SCORE_THRESH).float()).mean().item() if obj_num > 0. else 0.
        noobj_thresh_sum = (pre_obj[noobj_mask] > self.cfg.TEST.SCORE_THRESH).sum().item() / B
        # iou_sc = iou_xyxy(xywh2xyxy(pre_relative_box[indices]), labels[..., 2:6], type='N2N') if obj_num > 0. else 0.
        metrics['ob_l'] = obj_loss.item()
        metrics['nob_l'] = noobj_loss.item()
        metrics['cls_l'] = lcls.item()
        metrics['box_l'] = lbox.item()
        metrics['aware_l'] = loss_iou_aware.item()
        metrics['iou_sc'] = iou.mean()
        iou_percent = (iou > self.cfg.TEST.IOU_THRESH).sum() * 1.0 / len(iou)
        metrics['iou_p'] = iou_percent.item()
        metrics['obj_sc'] = obj_sc
        metrics['obj_p'] = obj_percent
        metrics['nob_sc'] = noobj_sc
        metrics['nob_t'] = noobj_thresh_sum

        # time_2 = time.time()
        # print('loss time LOSS CALL2-1:', time_2 - time_1)
        return total_loss, metrics

    def loss_cls(self, pre_cls, num_target, tcls, metrics):
        cp, cn = smooth_BCE(eps=0.0)
        t_cls = torch.full_like(pre_cls, cn, device=self.device)  # targets
        t_cls[range(num_target), tcls] = cp
        lcls = self.bceloss(pre_cls, t_cls)
        if self.reduction == 'sum':
            lcls = lcls / num_target  # BCE
        cls_score = ((pre_cls.max(-1)[1] == tcls).float()).mean()
        metrics['cls_p'] = cls_score.item()
        return lcls

    def Loss_Call(self, f_maps, dataset, kwargs):
        images, labels, datainfos = dataset
        labels[..., 2:6]=xyxy2xywh(labels[...,2:6])

        metrics = {}
        total_loss = torch.FloatTensor([0]).to(self.device)
        for f_id, f_map in enumerate(f_maps):
            _loss, _metrics = self._loss_cal_one_Fmap(f_map, f_id, labels, kwargs)
            total_loss += _loss
            if f_id == 0:
                metrics = _metrics
            else:
                for k, v in metrics.items():
                    try:
                        metrics[k] += _metrics[k]
                    except:
                        pass

        for k, v in metrics.items():
            metrics[k] = v / len(f_maps)
        return {'total_loss': total_loss, 'metrics': metrics}
