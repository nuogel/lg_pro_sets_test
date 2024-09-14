"""Loss calculation based on yolo."""
import torch
import numpy as np
from lgdet.util.util_iou import iou_xywh, xywh2xyxy, iou_xyxy
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
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.apollo_cls2idx = dict(zip(cfg.TRAIN.CLASSES, range(cfg.TRAIN.CLASSES_NUM)))
        self.parsepredict = ParsePredict(cfg)

        self.reduction = 'sum'
        self.mseloss = torch.nn.MSELoss(reduction=self.reduction)
        self.bceloss = torch.nn.BCELoss(reduction=self.reduction)
        self.alpha = 0.25
        self.gamma = 2
        self.Focalloss = FocalLoss(alpha=self.alpha, gamma=self.gamma)
        self.Focalloss_lg = FocalLoss_lg(alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)

    def _reshape_labels(self, pre_obj, labels, f_id):
        """
        Reshape the labels.

        :param labels: labels from training data
        :param grid_xy: the matrix of the grid numbers
        :return: lab_obj, lab_cls, lab_xy, lab_wh, lab_boxes, area_scal
        """
        B, C, H, W = pre_obj.shape
        mask = np.arange(self.anc_num) + self.anc_num * f_id  # be care of the relationship between anchor size and the feature map size.
        anchors_raw = self.anchors[mask]
        anchors = torch.Tensor([(a_w / self.cfg.TRAIN.IMG_SIZE[1] * W, a_h / self.cfg.TRAIN.IMG_SIZE[0] * H) for a_w, a_h in anchors_raw])

        # anchors = self.anchors[mask] / torch.Tensor([self.cfg.TRAIN.IMG_SIZE[1], self.cfg.TRAIN.IMG_SIZE[0]])  # * torch.Tensor([W, H])
        anchors = anchors.to(self.device)
        obj_mask = torch.BoolTensor(B, self.anc_num, H, W, ).fill_(0).to(self.device)
        noobj_mask = torch.BoolTensor(B, self.anc_num, H, W).fill_(1).to(self.device)
        lab_xy = torch.zeros([B, self.anc_num, H, W, 2]).to(self.device)
        lab_wh = torch.zeros([B, self.anc_num, H, W, 2]).to(self.device)
        lab_cls = torch.zeros([B, self.anc_num, H, W, self.cls_num]).to(self.device)
        grid_wh = torch.Tensor([W, H]).to(self.device)

        target_boxes = labels[..., 2:6]
        # x1y1x2y2->xywh
        gx1y1 = target_boxes[..., :2]
        gx2y2 = target_boxes[..., 2:]

        gxy = (gx1y1 + gx2y2) / 2.0 * grid_wh
        gwh = (gx2y2 - gx1y1) * grid_wh

        box_iou = torch.cat([torch.zeros_like(gwh), gwh], 1)
        anc = torch.cat([torch.zeros_like(anchors), anchors], 1)

        ious = iou_xywh(anc.to(self.device), box_iou.to(self.device), type='N2N')
        best_ious, best_n = ious.max(0)
        # Separate target values
        b, target_labels = labels[..., :2].long().t()
        gx, gy = gxy.t()
        gw, gh = gwh.t()
        gi, gj = gxy.long().t()
        # Set masks
        obj_mask[b, best_n, gj, gi] = 1
        noobj_mask[b, best_n, gj, gi] = 0
        ignore_thresh = 0.5
        for i, anchor_ious in enumerate(ious.t()):
            noobj_mask[b[i], anchor_ious > ignore_thresh, gj[i], gi[i]] = 0

        # Coordinates
        lab_xy[b, best_n, gj, gi, 0] = gx - gx.floor()
        lab_xy[b, best_n, gj, gi, 1] = gy - gy.floor()
        # Width and height
        lab_wh[b, best_n, gj, gi, 0] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
        lab_wh[b, best_n, gj, gi, 1] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)

        # One-hot encoding of label
        lab_cls[b, best_n, gj, gi, target_labels] = 1
        indx = [b, best_n, gj, gi]
        return obj_mask, noobj_mask, lab_cls, lab_xy, lab_wh, target_boxes, indx

    def _loss_cal_one_Fmap(self, f_map, f_id, labels, kwargs):
        """Calculate the loss."""
        # init loss.
        loc_losstype, obj_losstype = kwargs['losstype']
        metrics = {}

        pre_obj, pre_cls, pre_xy, pre_wh, pre_relative_box = self.parsepredict._parse_yolo_predict_fmap(f_map, f_id)
        obj_mask, noobj_mask, lab_cls, lab_xy, lab_wh, target_boxes, index = self._reshape_labels(pre_obj, labels, f_id)

        B, C, H, W = pre_obj.shape
        lab_obj = obj_mask.float()
        nub_obj = lab_obj.sum()

        if obj_losstype == 'focalloss':
            # # FOCAL loss
            _loss, obj_loss, noobj_loss = self.Focalloss_lg(pre_obj, lab_obj, obj_mask, noobj_mask, split_loss=True)
        elif obj_losstype == 'bce' or obj_losstype is None:
            # nomal loss
            obj_loss = self.bceloss(pre_obj[obj_mask], lab_obj[obj_mask])
            noobj_loss = self.bceloss(pre_obj[noobj_mask], lab_obj[noobj_mask])

        lxy_loss = self.mseloss(pre_xy[obj_mask], lab_xy[obj_mask])
        lwh_loss = self.mseloss(pre_wh[obj_mask], lab_wh[obj_mask])
        cls_loss = self.bceloss(pre_cls[obj_mask], lab_cls[obj_mask]) / (self.cls_num * nub_obj)
        loc_loss = lxy_loss + lwh_loss
        total_loss = (obj_loss + noobj_loss + loc_loss + cls_loss) / B

        cls_sc = (pre_cls[obj_mask].argmax(-1) == lab_cls[obj_mask].argmax(-1)).float().mean().item()
        obj_sc = pre_obj[obj_mask].mean().item()
        noobj_sc = pre_obj[noobj_mask].mean().item()
        obj_percent = ((pre_obj[obj_mask] > self.cfg.TEST.SCORE_THRESH).float()).mean().item()
        noobj_thresh_sum = (pre_obj[noobj_mask] > self.cfg.TEST.SCORE_THRESH).sum().item() / B
        iou_sc = iou_xyxy(xywh2xyxy(pre_relative_box[index]), target_boxes, type='N21').mean()
        metrics['cls_sc'] = cls_sc
        metrics['iou_sc'] = iou_sc
        metrics['obj_sc'] = obj_sc
        metrics['obj_p'] = obj_percent
        metrics['nob_sc'] = noobj_sc
        metrics['nob_t'] = noobj_thresh_sum
        metrics['ob_l'] = obj_loss.item()
        metrics['nob_l'] = noobj_loss.item()
        metrics['cls_l'] = cls_loss.item()
        metrics['box_l'] = loc_loss.item()

        return total_loss, metrics

    def _focal_loss(self, pred, target):  # not used
        ce = self.bceloss(pred, target)
        alpha = target * self.alpha + (1. - target) * (1. - self.alpha)
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_loss = alpha * (1. - pt) ** self.gamma * ce
        return focal_loss

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
