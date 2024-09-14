"""Loss calculation based on yolo."""
import torch
import math
import numpy as np
from lgdet.util.util_get_cls_names import _get_class_names
from lgdet.postprocess.parse_factory import ParsePredict

'''
with the new yolo loss, in 56 images, loss is 0.18 and map is 0.2.and the test show wrong bboxes.
with the new yolo loss, in 8 images, loss is 0.015 and map is 0.99.and the test show terrible bboxes.

'''


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou


def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


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

        self.alpha = 0.25
        self.gamma = 2
        self.use_hard_noobj_loss = False
        self.watch_metrics = self.cfg.TRAIN.WATCH_METIRICS

    # def _Loss_Call(self, p, dataset, kwargs):  # predictions, targets, model
    #     h = {'giou': 3.54,  # giou loss gain
    #          'cls': 37.4,  # cls loss gain
    #          'cls_pw': 1.0,  # cls BCELoss positive_weight
    #          'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
    #          'obj_pw': 1.0,  # obj BCELoss positive_weight
    #          'iou_t': 0.20,  # iou training threshold
    #          'lr0': 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)
    #          'lrf': 0.0005,  # final learning rate (with cos scheduler)
    #          'momentum': 0.937,  # SGD momentum
    #          'weight_decay': 0.0005,  # optimizer weight decay
    #          'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
    #          'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
    #          'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
    #          'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
    #          'degrees': 1.98 * 0,  # image rotation (+/- deg)
    #          'translate': 0.05 * 0,  # image translation (+/- fraction)
    #          'scale': 0.05 * 0,  # image scale (+/- gain)
    #          'shear': 0.641 * 0}  # image shear (+/- deg)
    #     gr = 1.0
    #     images, targets, datainfos = dataset
    #
    #     ft = torch.cuda.FloatTensor if p[0].is_cuda else torch.Tensor
    #     lcls, lbox, lobj = ft([0]), ft([0]), ft([0])
    #     tcls, tbox, indices, anchors = self._build_targets(p, targets, h)  # targets
    #     # hyperparameters
    #     red = 'mean'  # Loss reduction (sum or mean)
    #
    #     # Define criteria
    #     BCEcls = nn.BCEWithLogitsLoss(pos_weight=ft([h['cls_pw']]), reduction=red)
    #     BCEobj = nn.BCEWithLogitsLoss(pos_weight=ft([h['obj_pw']]), reduction=red)
    #
    #     # class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
    #     cp, cn = smooth_BCE(eps=0.0)
    #
    #     # per output
    #     nt = 0  # targets
    #     for i, pi in enumerate(p):  # layer index, layer predictions
    #         B, C, H, W = pi.shape
    #         pi = pi.view(B, self.anc_num, self.cls_num + 5, H, W)
    #         pi = pi.permute(0, 1, 3, 4, 2).contiguous()  # ((0, 1, 3, 4, 2),   (0, 3, 4, 1, 2))
    #
    #         b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
    #         tobj = torch.zeros_like(pi[..., 0])  # target obj
    #
    #         nb = b.shape[0]  # number of targets
    #         if nb:
    #             nt += nb  # cumulative targets
    #             ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
    #
    #             giou = 0
    #             if giou:
    #                 # GIoU
    #                 pxy = ps[:, :2].sigmoid()
    #                 pwh = ps[:, 2:4].exp().clamp(max=1E3) * anchors[i]
    #                 pbox = torch.cat((pxy, pwh), 1)  # predicted box
    #                 giou = bbox_iou(pbox.t(), tbox[i], x1y1x2y2=False, GIoU=True)  # giou(prediction, target)
    #                 lbox += (1.0 - giou).sum() if red == 'sum' else (1.0 - giou).mean()  # giou loss
    #                 tobj[b, a, gj, gi] = (1.0 - gr) + gr * giou.detach().clamp(0).type(tobj.dtype)  # giou ratio
    #             else:
    #                 # mse:
    #                 pxy = ps[:, :2].sigmoid()
    #                 pwh = ps[:, 2:4]
    #                 twh = torch.log(tbox[i][..., 2:] / anchors[i] + 1e-16)  # torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    #                 lxy = self.mseloss(pxy, tbox[i][..., :2])
    #                 lwh = self.mseloss(pwh, twh)
    #                 lbox += lxy + lwh
    #                 tobj[b, a, gj, gi] = 1.0
    #
    #             # Class
    #             if self.cls_num > 1:  # cls loss (only if multiple classes)
    #                 t = torch.full_like(ps[:, 5:], cn)  # targets
    #                 t[range(nb), tcls[i]] = cp
    #                 lcls += BCEcls(ps[:, 5:], t)  # BCE
    #
    #         lobj += BCEobj(pi[..., 4], tobj)  # obj loss
    #
    #     return {'obj_loss': lobj, 'cls_loss': lcls, 'loc_loss': lbox}

    # def _build_targets(self, p, labels, h):
    #     # x1y1x2y2 to xywh:
    #     targets = labels.clone()
    #     targets[..., 2:4] = (labels[..., 2:4] + labels[..., 4:6]) / 2
    #     targets[..., 4:6] = (labels[..., 4:6] - labels[..., 2:4])
    #
    #     # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    #     nt = targets.shape[0]
    #     tcls, tbox, indices, anch = [], [], [], []
    #     gain = torch.ones(6, device=targets.device)  # normalized to gridspace gain
    #     # off = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float()  # overlap offsets
    #
    #     for i, pi in enumerate(p):
    #         B, C, H, W = pi.shape
    #         mask = np.arange(self.anc_num) + self.anc_num * i  # be care of the relationship between anchor size and the feature map size.
    #         anchors_raw = self.anchors[mask]
    #         anchors = torch.Tensor([(a_w / self.cfg.TRAIN.IMG_SIZE[1] * W, a_h / self.cfg.TRAIN.IMG_SIZE[0] * H) for a_w, a_h in anchors_raw]).to(self.device)
    #
    #         gain[2:] = torch.tensor([W, H, W, H])  # xyxy gain
    #         na = anchors.shape[0]  # number of anchors
    #         at = torch.arange(na).view(na, 1).repeat(1, nt)  # anchor tensor, same as .repeat_interleave(nt)
    #
    #         # Match targets to anchors
    #         a, t, offsets = [], targets * gain, 0
    #         if nt:
    #             j = wh_iou(anchors, t[:, 4:6]) > 0.2  # iou(3,n) = wh_iou(anchors(3,2), gwh(n,2))
    #             a, t = at[j], t.repeat(na, 1, 1)[j]  # filter
    #         # Define
    #         b, c = t[:, :2].long().T  # image, class
    #         gxy = t[:, 2:4]  # grid xy
    #         gwh = t[:, 4:6]  # grid wh
    #         gij = (gxy - offsets).long()
    #         gi, gj = gij.T  # grid xy indices
    #
    #         # Append
    #         indices.append((b, a, gj, gi))  # image, anchor, grid indices
    #         tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
    #         anch.append(anchors[a])  # anchors
    #         tcls.append(c)  # class
    #         if c.shape[0]:  # if any targets
    #             assert c.max() < self.cls_num, 'Model accepts %g classes labeled from 0-%g, however you labelled a class %g. ' \
    #                                            'See https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data' % (
    #                                                self.cls_num, self.cls_num - 1, c.max())
    #     return tcls, tbox, indices, anch

    def build_targets(self, pre_obj, labels, f_id):
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
            j = wh_iou(anchors, t[:, 4:6]) > 0.2  # iou(3,n) = wh_iou(anchors(3,2), gwh(n,2))
            a, t = at[j], t.repeat(self.anc_num, 1, 1)[j]  # filter
        # Define
        batch, tcls = t[:, :2].long().T  # image, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = gxy.long()
        gi, gj = gij.T  # grid xy indices

        indices = [batch, a, gj, gi]  # image, anchor, grid indices
        tbox = torch.cat((gxy - gij, gwh), 1)  # box
        anch = anchors[a]  # anchors
        if tcls.shape[0]:  # if any targets
            assert tcls.max() < self.cls_num, 'cls_num is beyound the classes.'
        return tcls, tbox, indices, anch

    def _loss_cal_one_Fmap(self, f_map, f_id, labels, losstype=None):
        """Calculate the loss."""
        # init loss.
        lcls, lxy, lwh, lobj = [torch.FloatTensor([0]).to(self.device) for _ in range(4)]

        pre_obj, pre_cls, pre_loc_xy, pre_loc_wh = self.parsepredict._parse_yolo_predict_fmap(f_map, f_id)
        tcls, tbox, indices, anchors = self.build_targets(pre_obj, labels, f_id)  # targets

        tobj = torch.zeros_like(pre_obj)  # target obj
        tobj[indices] = 1.0
        nb = indices[0].shape[0]  # number of targets
        if nb:
            pre_cls, pre_xy, pre_wh = [i[indices] for i in [pre_cls, pre_loc_xy, pre_loc_wh]]

            # mse:
            twh = torch.log(tbox[..., 2:] / anchors + 1e-16)  # torch.log(gw / anchors[best_n][:, 0] + 1e-16)
            lxy = self.mseloss(pre_xy, tbox[..., :2])
            lwh = self.mseloss(pre_wh, twh)

            t_cls = torch.zeros_like(pre_cls)  # targets
            t_cls[range(nb), tcls] = 1
            lcls = self.bceloss(pre_cls, t_cls)  # BCE

        lobj = self.bceloss(pre_obj, tobj)  # obj loss

        total_loss = lobj + lcls + lxy + lwh
        metrics = {'obj_loss': [lobj.item()],
                   'cls_loss': [lcls.item()],
                   'xy_loss': [lcls.item()],
                   'wh_loss': [lcls.item()],
                   }
        return total_loss, metrics

    def Loss_Call(self, f_maps, dataset, kwargs):
        losstype = kwargs['losstype']
        images, labels, datainfos = dataset
        metrics = {}
        total_loss = torch.FloatTensor([0]).to(self.device)
        for f_id, f_map in enumerate(f_maps):
            _loss, _metrics = self._loss_cal_one_Fmap(f_map, f_id, labels, losstype)
            total_loss += _loss
            if f_id == 0:
                metrics = _metrics
            else:
                for k, v in metrics.items():
                    metrics[k].append(_metrics[k][0])

        for k, v in metrics.items():
            metrics[k] = '%.2f' % np.asarray(v, np.float32).mean()

        return {'total_loss': total_loss, 'metrics': metrics}
