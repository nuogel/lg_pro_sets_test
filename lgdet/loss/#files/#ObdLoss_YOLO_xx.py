"""Loss calculation based on yolo."""
import torch
import numpy as np
from lgdet.util.util_iou import iou_xywh, xywh2xyxy, iou_xyxy
from lgdet.util.util_get_cls_names import _get_class_names
from lgdet.postprocess.parse_factory import ParsePredict

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

        self.mseloss = torch.nn.MSELoss()
        self.bceloss = torch.nn.BCELoss()
        self.class_name = _get_class_names(cfg.PATH.CLASSES_PATH)

        self.parsepredict = ParsePredict(cfg)
        self.multiply_area_scale = 0  # whether multiply loss to area_scale.

        self.alpha = 0.5
        self.gamma = 2
        self.use_hard_noobj_loss = False
        self.watch_metrics = self.cfg.TRAIN.WATCH_METIRICS

    def _reshape_labels(self, pre_obj, pre_cls, pre_loc_xy, pre_loc_wh, labels, f_id):
        """
        Reshape the labels.

        :param labels: labels from training data
        :param grid_xy: the matrix of the grid numbers
        :return: labels_obj, labels_cls, lab_loc_xy, lab_loc_wh, labels_boxes, area_scal
        """
        B, C, H, W = pre_obj.shape
        mask = np.arange(self.anc_num) + self.anc_num * f_id  # be care of the relationship between anchor size and the feature map size.
        anchors_raw = self.anchors[mask]
        anchors = torch.Tensor([(a_w / self.cfg.TRAIN.IMG_SIZE[1] * W, a_h / self.cfg.TRAIN.IMG_SIZE[0] * H) for a_w, a_h in anchors_raw])

        # anchors = self.anchors[mask] / torch.Tensor([self.cfg.TRAIN.IMG_SIZE[1], self.cfg.TRAIN.IMG_SIZE[0]])  # * torch.Tensor([W, H])
        anchors = anchors.to(self.device)
        obj_mask = torch.BoolTensor(B, self.anc_num, H, W, ).fill_(0).to(self.device)
        noobj_mask = torch.BoolTensor(B, self.anc_num, H, W).fill_(1).to(self.device)
        labels_loc_xy = torch.zeros([B, self.anc_num, H, W, 2]).to(self.device)
        labels_loc_wh = torch.zeros([B, self.anc_num, H, W, 2]).to(self.device)
        labels_cls = torch.zeros([B, self.anc_num, H, W, self.cls_num]).to(self.device)

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
        labels_loc_xy[b, best_n, gj, gi, 0] = gx - gx.floor()
        labels_loc_xy[b, best_n, gj, gi, 1] = gy - gy.floor()
        # Width and height
        labels_loc_wh[b, best_n, gj, gi, 0] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
        labels_loc_wh[b, best_n, gj, gi, 1] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
        # One-hot encoding of label
        labels_cls[b, best_n, gj, gi, target_labels] = 1

        if self.watch_metrics:
            class_mask = torch.zeros([B, self.anc_num, H, W]).to(self.device)
            iou_scores = torch.zeros([B, self.anc_num, H, W]).to(self.device)
            # # Compute label correctness and iou at best anchor
            pre_loc = torch.cat((pre_loc_xy, pre_loc_wh), -1)
            class_mask[b, best_n, gj, gi] = (pre_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()

            # grid_x = torch.arange(0, W).view(-1, 1).repeat(1, H).unsqueeze(2).permute(1, 0, 2)
            # grid_y = torch.arange(0, H).view(-1, 1).repeat(1, W).unsqueeze(2)
            # grid_xy = torch.cat([grid_x, grid_y], 2).unsqueeze(2).unsqueeze(0). \
            #     expand(1, self.anc_num, H, W, 2).expand_as(pre_loc_xy).to(self.device)
            grid_x = torch.arange(W).repeat(H, 1).view([1, 1, H, W]).to(self.device)
            grid_y = torch.arange(H).repeat(W, 1).t().view([1, 1, H, W]).to(self.device)

            pre_realtive_xy = pre_loc_xy.clone()
            pre_realtive_xy[...,0] = pre_loc_xy[...,0] + grid_x
            pre_realtive_xy[...,1] = pre_loc_xy[...,1] + grid_y
            # prepare gird xy
            pre_realtive_xy = pre_realtive_xy / grid_wh

            anchor_ch = anchors.view(1, self.anc_num,1, 1,  2).expand(1, self.anc_num, H, W, 2).to(self.device)
            pre_wh = pre_loc_wh.exp() * anchor_ch
            pre_realtive_wh = pre_wh / grid_wh

            pre_relative_box = torch.cat([pre_realtive_xy, pre_realtive_wh], -1)

            iou_scores[b, best_n, gj, gi] = iou_xyxy(xywh2xyxy(pre_relative_box[b, best_n, gj, gi]).to(self.device),
                                                     target_boxes.to(self.device), type='N21')

            cls_acc = class_mask[obj_mask].mean()
            conf_obj = pre_obj[obj_mask].mean()
            conf_noobj = pre_obj[noobj_mask].mean()
            conf50 = (pre_obj > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * obj_mask.float()
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            metrics = {
                "cls_acc": cls_acc.item(),
                "recall50": recall50.item(),
                "recall75": recall75.item(),
                "precision": precision.item(),
                "conf_obj": conf_obj.item(),
                "conf_noobj": conf_noobj.item(),
            }

        if self.multiply_area_scale:
            # TODO: labels_loc_wh is no fit for area_scale
            area_scale = (1 - torch.sqrt(labels_loc_wh[..., 0] * labels_loc_wh[..., 1])).unsqueeze(-1).expand_as(
                labels_loc_wh)
        else:
            area_scale = 1.0
        return obj_mask, noobj_mask, labels_cls, labels_loc_xy, labels_loc_wh, metrics

    def _hard_noobj_mask(self, pre_obj, noobj_mask, obj_mask, hard_num=1000):  # TODO: obj_mask==8 & noobj_mask == 12220
        '''
        due to the imbalance betwine  obj and noobj, (obj=10, noobj=500000).
        so, we design the hard_noobj function to find the top N noobj to backward.
        :return: noobj mask.
        PS:
            When training coco. the loss is steady,and won't go down...
        '''
        # mask = torch.ge(pre_obj, thresh_score)
        # mask = mask & noobj_mask
        pre_obj_copy = pre_obj.clone()

        pre_obj_copy[obj_mask] = 0.

        nomask = pre_obj_copy > 0.2

        if torch.sum(noobj_mask) == 0:
            nomask = noobj_mask
        return nomask

    def _loss_cal_one_Fmap(self, f_map, f_id, labels, losstype=None):
        """Calculate the loss."""

        pre_obj, pre_cls, pre_loc_xy, pre_loc_wh = self.parsepredict._parse_yolo_predict_fmap(f_map, f_id)
        obj_mask, noobj_mask, labels_cls, labels_loc_xy, labels_loc_wh, metrics = self._reshape_labels(pre_obj, pre_cls, pre_loc_xy, pre_loc_wh, labels, f_id)

        labels_obj = obj_mask.float()
        if self.use_hard_noobj_loss:
            noobj_mask = self._hard_noobj_mask(pre_obj, noobj_mask, obj_mask)

        if losstype == 'focalloss':
            # # FOCAL loss
            obj_loss = (self.alpha * (((1. - pre_obj)[obj_mask]) ** self.gamma)) * self.bceloss(pre_obj[obj_mask], labels_obj[obj_mask])
            noobj_loss = ((1 - self.alpha) * ((pre_obj[noobj_mask]) ** self.gamma)) * self.bceloss(pre_obj[noobj_mask], labels_obj[noobj_mask])
            obj_loss = torch.mean(obj_loss)
            noobj_loss = torch.mean(noobj_loss)

        elif losstype == 'bce' or losstype is None:
            # nomal loss
            obj_loss = self.bceloss(pre_obj[obj_mask], labels_obj[obj_mask])
            noobj_loss = 100 * self.bceloss(pre_obj[noobj_mask], labels_obj[noobj_mask])
        else:
            print(losstype, 'is not define.')
            obj_loss = 0.
            noobj_loss = 0.
        lxy_loss = self.mseloss(pre_loc_xy[obj_mask], labels_loc_xy[obj_mask])
        lwh_loss = self.mseloss(pre_loc_wh[obj_mask], labels_loc_wh[obj_mask])
        cls_loss = self.bceloss(pre_cls[obj_mask], labels_cls[obj_mask])
        loc_loss = lxy_loss + lwh_loss
        return obj_loss, noobj_loss, cls_loss, loc_loss, metrics

    def _focal_loss(self, pred, target):  # not used
        ce = self.bceloss(pred, target)
        alpha = target * self.alpha + (1. - target) * (1. - self.alpha)
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_loss = alpha * (1. - pt) ** self.gamma * ce
        return focal_loss

    def Loss_Call(self, f_maps, dataset, kwargs):
        losstype = kwargs['losstype']
        images, labels, datainfos = dataset
        noobj_loss, obj_loss, cls_loss, loc_loss = 0.0, 0.0, 0.0, 0.0
        metrics = {}
        for f_id, f_map in enumerate(f_maps):
            _obj_loss, _noobj_loss, _cls_loss, _loc_loss, _metrics = self._loss_cal_one_Fmap(f_map, f_id, labels, losstype)
            obj_loss += _obj_loss
            noobj_loss += _noobj_loss
            cls_loss += _cls_loss
            loc_loss += _loc_loss
            if f_id == 0:
                metrics = _metrics
            else:
                for k, v in metrics.items():
                    metrics[k] += _metrics[k]

        for k, v in metrics.items():
            metrics[k] = '%.3f' % (v / len(f_maps))

        return {'obj_loss': obj_loss, 'noobj_loss': noobj_loss, 'cls_loss': cls_loss, 'loc_loss': loc_loss, 'metrics': metrics}
