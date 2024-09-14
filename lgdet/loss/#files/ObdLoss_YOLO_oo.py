"""Loss calculation based on yolo."""
import torch
import numpy as np
from lgdet.util.util_iou import _iou_wh, bbox_GDCiou
from lgdet.postprocess.parse_factory import ParsePredict

from lgdet.util.util_loss import FocalLoss, FocalLoss_lg

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

        self.reduction = 'sum'
        self.mseloss = torch.nn.MSELoss(reduction=self.reduction)
        self.bceloss = torch.nn.BCELoss(reduction=self.reduction)

        self.parsepredict = ParsePredict(cfg)
        self.multiply_area_scale = 0  # whether multiply loss to area_scale.

        self.alpha = 0.25
        self.gamma = 2
        self.Focalloss = FocalLoss(alpha=self.alpha, gamma=self.gamma)
        self.Focalloss_lg = FocalLoss_lg(alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)
        self.multiply_area_scale = 0

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
        gain = torch.ones(6, device=targets.device)  # normalized to gridspace gain
        gain[2:] = torch.tensor([W, H, W, H])  # xyxy gain
        t = targets * gain

        self.anc_num = anchors.shape[0]  # number of anchors
        at = torch.arange(self.anc_num).view(self.anc_num, 1).repeat(1, num_target)  # anchor tensor, same as .repeat_interleave(num_target)

        # Match targets to anchors
        if num_target:
            ious = _iou_wh(anchors, t[:, 4:6])
            a = ious.max(dim=0)[1]  # lg
            j = ious > 0.5  # iou(3,n) = wh_iou(anchors(3,2), gwh(n,2))
            a_ignore, t_ignore = at[j], t.repeat(self.anc_num, 1, 1)[j]  # filter
        # Define
        batch, tcls = t[:, :2].long().t()  # image, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = gxy.long()
        gi, gj = gij.t()  # grid xy indices
        indices = [batch, a, gj, gi]  # image, anchor, grid indices

        batch_ignore, tcls_ignore = t_ignore[:, :2].long().t()  # image, class
        gxy_ignore = t_ignore[:, 2:4]  # grid xy
        gij_ignore = gxy_ignore.long()
        gi_ignore, gj_ignore = gij_ignore.t()  # grid xy indices
        indices_ignore = [batch_ignore, a_ignore, gj_ignore, gi_ignore]  # image, anchor, grid indices

        tbox = torch.cat((gxy - gij.float(), gwh), 1)  # box
        anch = anchors[a]  # anchors
        if tcls.shape[0]:  # if any targets
            assert tcls.max() < self.cls_num, 'cls_num is beyound the classes.'
        # time_2 = time.time()
        # print('build target time LOSS CALL:', time_2 - time_1)
        return tcls, tbox, indices, indices_ignore, anch

    def _loss_cal_one_Fmap(self, f_map, f_id, labels, kwargs):
        """Calculate the loss."""
        # init loss.
        loc_losstype, obj_losstype = kwargs['losstype']

        metrics = {}
        lcls, lbox, lobj = [torch.FloatTensor([0]).to(self.device) for _ in range(3)]

        pre_obj, pre_cls, pre_loc_xy, pre_loc_wh = self.parsepredict._parse_yolo_predict_fmap(f_map, f_id)
        tcls, tbox, indices, indices_ignore, anchors = self.build_targets(pre_obj, labels, f_id)  # targets
        B, C, H, W = pre_obj.shape
        tobj = torch.zeros_like(pre_obj)  # target obj
        num_target = anchors.shape[0]  # number of targets
        loss_ratio = {'box': 1, 'cls': 1, 'obj': 1, 'noobj': 1}
        if num_target:
            pre_cls, pre_xy, pre_wh = [i[indices] for i in [pre_cls, pre_loc_xy, pre_loc_wh]]
            t_cls = torch.zeros_like(pre_cls)  # targets
            t_cls[range(num_target), tcls] = 1
            lcls = self.bceloss(pre_cls, t_cls)
            if self.reduction == 'sum':
                lcls = (lcls / num_target) / self.cls_num  # BCE
            cls_score = ((pre_cls.max(-1)[1] == tcls).float()).mean()
            metrics['cls_p'] = cls_score.item()

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
                # metrics
                _pre_wh = pre_wh.exp() * anchors
                pbox = torch.cat((pre_xy, _pre_wh), 1)  # predicted box
                iou = bbox_GDCiou(pbox.t(), tbox, x1y1x2y2=False)  # giou(prediction, target)
                metrics['iou'] = iou.mean().item()
                iou_percent = (iou > self.cfg.TEST.IOU_THRESH).sum() * 1.0 / len(iou)
                metrics['iou_p'] = iou_percent.item()

            elif loc_losstype == 'giou':
                pre_wh = pre_wh.exp().clamp(max=1E3) * anchors
                pbox = torch.cat((pre_xy, pre_wh), 1)  # predicted box
                giou = bbox_GDCiou(pbox.t(), tbox, x1y1x2y2=False, GIoU=True)  # giou(prediction, target)
                lbox = (1.0 - giou) * area_scale
                if self.reduction == 'mean':
                    lbox = lbox.mean()  # giou loss
                else:
                    lbox = lbox.sum() / num_target  # giou loss
                # Obj
                global_step = kwargs['global_step']
                if self.one_test:
                    gr = 1
                else:
                    len_batch = kwargs['len_batch']
                    xi = [0, max(3 * len_batch, 500)]
                    gr = np.interp(global_step, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                tobj[indices] = (1.0 - gr) + gr * giou.detach().clamp(min=0.1).type(tobj.dtype)  # giou ratio
                metrics['giou'] = giou.mean().item()
                iou_percent = (giou > self.cfg.TEST.IOU_THRESH).sum() * 1.0 / len(giou)
                metrics['iou_p'] = iou_percent.item()

            else:
                print('sorry, no such a loss type...')
                exit()


        else:
            pass
        obj_mask = tobj > 0.
        noobj_mask = ~obj_mask
        noobj_mask[indices_ignore] = False
        obj_num = obj_mask.sum()
        if obj_losstype == 'focalloss':
            loss_ratio['obj'] = 1
            loss_ratio['noobj'] = 1
            _loss, obj_loss, noobj_loss = self.Focalloss_lg(pre_obj, tobj, obj_mask, noobj_mask, split_loss=True)

        elif obj_losstype == 'bce':
            loss_ratio['obj'] = 1
            loss_ratio['noobj'] = 1
            if obj_num:
                obj_loss = self.bceloss(pre_obj[obj_mask], tobj[obj_mask])  # obj loss
            else:
                obj_loss = torch.FloatTensor([0]).to(self.device)
            noobj_loss = self.bceloss(pre_obj[noobj_mask], tobj[noobj_mask])  # obj loss

        elif obj_losstype == 'mse':
            loss_ratio['obj'] = 1
            loss_ratio['noobj'] = 5
            if obj_num:
                obj_loss = self.mseloss(pre_obj[obj_mask], tobj[obj_mask])  # obj loss
            else:
                obj_loss = torch.FloatTensor([0]).to(self.device)
            noobj_loss = self.mseloss(pre_obj[noobj_mask], tobj[noobj_mask])  # obj loss

        total_loss = loss_ratio['obj'] * obj_loss + loss_ratio['noobj'] * noobj_loss + loss_ratio['cls'] * lcls + loss_ratio['box'] * lbox

        if self.reduction == 'sum': total_loss = total_loss / B
        if torch.isnan(total_loss) or total_loss.item() == float("inf") or total_loss.item() == -float("inf"):
            print('nan')
        obj_sc = pre_obj[obj_mask].mean().item() if obj_num > 0. else 0.
        noobj_sc = pre_obj[noobj_mask].mean().item()
        obj_percent = ((pre_obj[obj_mask] > self.cfg.TEST.SCORE_THRESH).float()).mean().item() if obj_num > 0. else 0.
        noobj_thresh_sum = (pre_obj[noobj_mask] > self.cfg.TEST.SCORE_THRESH).sum().item() / B
        metrics['obj_sc'] = obj_sc
        metrics['obj_p'] = obj_percent
        metrics['nob_sc'] = noobj_sc
        metrics['nob_t'] = noobj_thresh_sum
        metrics['ob_l'] = obj_loss.item()
        metrics['nob_l'] = noobj_loss.item()
        metrics['cls_l'] = lcls.item()
        metrics['box_l'] = lbox.item()

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
