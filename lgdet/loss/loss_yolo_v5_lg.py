"""Loss calculation based on yolo."""
import torch
import numpy as np
from lgdet.util.util_iou import _iou_wh, bbox_GDCiou, xywh2xyxy, iou_xyxy, wh_iou
from lgdet.postprocess.parse_factory import ParsePredict
import torch.nn.functional as F

from lgdet.loss.loss_base.focal_loss import FocalLoss, FocalLoss_lg
from lgdet.loss.loss_base.ghm_loss import GHMC
from lgdet.loss.loss_base.smothBCE import smooth_BCE

'''
some bugs for not good results...
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
        self.Focalloss = FocalLoss(gamma=2, alpha=0.75, add_logist=True, reduction='mean')
        self.Focalloss_lg = FocalLoss_lg(alpha=self.alpha, gamma=self.gamma, )
        self.ghm = GHMC(use_sigmoid=True)

    def build_targets(self, pre_obj, labels, f_id):
        # time_1 = time.time()

        # x1y1x2y2 to xywh:
        targets = labels.clone()
        targets[..., 2:4] = (labels[..., 2:4] + labels[..., 4:6]) / 2
        targets[..., 4:6] = (labels[..., 4:6] - labels[..., 2:4])
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
            # j = wh_iou(anchors, t[:,:, 4:6]) > 0.2  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
            add_maxIou = 0  # by:lg
            if add_maxIou:
                iou = wh_iou(anchors, t[0, :, 4:6]).max(0)[1]
                fited = j.max(0)[0].detach()
                for i, fi in enumerate(fited):
                    if fi == False:
                        j[iou[i], i] = True

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
        tcls = c
        # self.check_2_bbox_in_one_grid(tcls.clone(), indices.copy(), tbox)
        return tcls, tbox, indices, anch

    def check_2_bbox_in_one_grid(self, tcls, indices, tbox):
        indices.append(tcls)
        all = torch.stack(indices).T
        same = 0
        for i in range(len(all) - 1):
            for j in range(i + 1, len(all)):
                if (all[i] == all[j]).sum() == 5:
                    same += 1
                    print(same, '-same:', str(i), '-', str(j), all[i], all[j], tbox[i], tbox[j])

    def _loss_cal_one_Fmap(self, f_map, f_id, labels, kwargs):
        """Calculate the loss."""
        # init loss.
        lcls, lbox, loss_iou_aware = [torch.FloatTensor([0.]).to(self.device) for _ in range(3)]
        iou, iou_percent, cls_p = [torch.FloatTensor([-1.]).to(self.device) for _ in range(3)]
        loc_losstype, obj_losstype = kwargs['losstype']
        balance = [4.0, 1.0, 0.4][f_id]
        metrics = {}
        pred_conf, pred_cls, pred_xy, pred_wh, pred_iou, pre_relative_box = self.parsepredict.parser._parse_yolo_predict_fmap(f_map, f_id)
        pre_obj, pre_cls, pre_loc_xy, pre_loc_wh = pred_conf.value, pred_cls.value, pred_xy.value, pred_wh.value

        with torch.no_grad():
            global_step = kwargs['global_step']
            if self.one_test:
                gr = 1
                eps = 0.
            else:
                len_batch = kwargs['len_batch']
                xi = [0, max(3 * len_batch, 500)]
                xj = [0, max(10 * len_batch, 500)]
                gr = np.interp(global_step, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                eps = np.interp(global_step, xj, [0.2, 0.0])
            tcls, tbox, indices, anchors = self.build_targets(pre_obj, labels, f_id)  # targets
            B, C, H, W = pre_obj.shape
            cp, cn = smooth_BCE(eps=eps)
            tobj = torch.full_like(pre_obj, cn, device=self.device)  # targets
            obj_mask = torch.zeros_like(pre_obj).type(torch.BoolTensor)  # targets
            obj_mask[indices] = True
            noobj_mask = ~obj_mask
            num_target = anchors.shape[0]  # number of targets
            loss_ratio = {'box': 0.05, 'cls': 0.125, 'obj': 10 * balance, 'noobj': 1 * balance}
        if num_target:
            pre_cls, pre_xy, pre_wh = [i[indices] for i in [pre_cls, pre_loc_xy, pre_loc_wh]]
            cp, cn = smooth_BCE(eps=eps)
            # t_cls = torch.full_like(pre_cls, cn, device=self.device)  # targets
            # t_cls[range(num_target), tcls] = cp
            # lcls = self.bceloss(pre_cls, t_cls)
            assert pred_cls.sigmoid_tag is False
            lcls = F.cross_entropy(pre_cls, tcls.type(torch.LongTensor).cuda())
            if self.reduction == 'sum':
                lcls = lcls / num_target  # BCE
            cls_p = ((pre_cls.max(-1)[1] == tcls).float()).mean()

            # 1) boxes loss.
            if self.multiply_area_scale:
                area = tbox[..., 2] * tbox[..., 3] / (H * W)
                area_scale = (1 - torch.sqrt(area) * 2).clamp(0.2, 1)  # 2 is scale parameter
            else:
                area_scale = 1.

            if loc_losstype == 'iouloss':
                pre_wh = pre_wh * anchors
                pbox = torch.cat((pre_xy, pre_wh), 1)  # predicted box
                iou = bbox_GDCiou(pbox.t(), tbox, x1y1x2y2=False, CIoU=True)  # giou(prediction, target)
                # if f_id == 0:
                #     print('\n', pbox[0], '\n', tbox[0])
                ratio_iou = False
                if ratio_iou:
                    riou = torch.ones_like(iou)  # IOU ratio
                    riou[iou < 0.5] = 5
                else:
                    riou = 1.0
                lbox = (1.0 - iou) * area_scale * riou
                if self.reduction == 'mean':
                    lbox = lbox.mean()  # giou loss
                else:
                    lbox = lbox.sum() / num_target  # giou loss

                # 2) Objectness
                tobj[indices] = (1.0 - gr) + gr * iou.detach().clamp(min=0.1).type(tobj.dtype)  # giou ratio

                if self.cfg.TRAIN.IOU_AWARE:
                    iou = torch.clamp(iou, 0, 1)
                    pred_iou = torch.clamp(pred_iou[indices], 0, 1)
                    loss_iou_aware = self.bceloss(pred_iou, iou.detach())
                iou_percent = (iou > self.cfg.TEST.IOU_THRESH).sum() * 1.0 / len(iou)
        else:
            pass
        label_weight_mask = (obj_mask | noobj_mask)
        obj_num = obj_mask.sum()
        if obj_losstype == 'focalloss':
            _loss, obj_loss, noobj_loss = self.Focalloss(pred_conf.value_raw, tobj, obj_mask=obj_mask, noobj_mask=noobj_mask)
            if obj_loss is None or noobj_loss is None:
                obj_loss = _loss / 2.
                noobj_loss = _loss / 2.
        elif obj_losstype == 'ghm':
            obj_loss = self.ghm(pre_obj, tobj, label_weight_mask)
            noobj_loss = torch.FloatTensor([0]).to(self.device)
        elif obj_losstype == 'bce':
            if obj_num:
                obj_loss = self.bceloss(pre_obj[obj_mask], tobj[obj_mask])  # obj loss
            else:
                obj_loss = torch.FloatTensor([0]).to(self.device)
            noobj_loss = self.bceloss(pre_obj[noobj_mask], tobj[noobj_mask])  # obj loss
        else:  # obj_losstype == 'mse':
            if obj_num:
                obj_loss = self.mseloss(pre_obj[obj_mask], tobj[obj_mask])  # obj loss
            else:
                obj_loss = torch.FloatTensor([0]).to(self.device)
            noobj_loss = self.mseloss(pre_obj[noobj_mask], tobj[noobj_mask])  # obj loss

        obj_loss = loss_ratio['obj'] * obj_loss
        noobj_loss = loss_ratio['noobj'] * noobj_loss
        lcls = loss_ratio['cls'] * lcls
        lbox = loss_ratio['box'] * lbox
        loss_iou_aware = loss_ratio['box'] * loss_iou_aware
        total_loss = obj_loss + noobj_loss + lcls + lbox + loss_iou_aware
        if self.reduction == 'sum': total_loss = total_loss / B
        if torch.isnan(total_loss) or total_loss.item() == float("inf") or total_loss.item() == -float("inf"):
            print('nan')

        # metrics
        # pre_obj = pre_obj.sigmoid()
        obj_sc = pre_obj[obj_mask].mean().item() if obj_num > 0. else -1.
        noobj_sc = pre_obj[noobj_mask].mean().item()
        obj_percent = ((pre_obj[obj_mask] > self.cfg.TEST.SCORE_THRESH).float()).mean().item() if obj_num > 0. else -1.
        noobj_thresh_sum = (pre_obj[noobj_mask] > self.cfg.TEST.SCORE_THRESH).sum().item() / B
        # iou_sc = iou_xyxy(xywh2xyxy(pre_relative_box[indices]), labels[..., 2:6], type='N2N') if obj_num > 0. else 0.
        metrics['cls_p'] = cls_p.item()
        metrics['iou_sc'] = torch.clamp(iou, 0, 1).mean().item()
        metrics['iou_p'] = iou_percent.item()
        metrics['obj_sc'] = obj_sc
        metrics['obj_p'] = obj_percent
        metrics['nob_sc'] = noobj_sc
        metrics['nob_t'] = noobj_thresh_sum
        metrics['ob_l'] = obj_loss.item()
        metrics['nob_l'] = noobj_loss.item()
        metrics['cls_l'] = lcls.item()
        metrics['box_l'] = lbox.item()
        metrics['aware_l'] = loss_iou_aware.item()



        # time_2 = time.time()
        # print('loss time LOSS CALL2-1:', time_2 - time_1)
        return total_loss, metrics

    def Loss_Call(self, f_maps, dataset, kwargs):
        images, labels, datainfos = dataset
        metrics = {}
        total_loss = torch.FloatTensor([0]).to(self.device)
        for f_id, f_map in enumerate(f_maps):
            _loss, _metrics = self._loss_cal_one_Fmap(f_map, f_id, labels, kwargs)
            total_loss += _loss
            for k, v in _metrics.items():
                if k not in metrics:
                    metrics[k] = []
                try:
                    if _metrics[k] < 0.:
                        pass
                    else:
                        metrics[k].append(_metrics[k])
                except:
                    pass

        for k, v in metrics.items():
            metrics[k] = np.asarray(v).mean()
        return {'total_loss': total_loss, 'metrics': metrics}
