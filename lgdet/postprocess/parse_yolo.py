"""Parse the predictions."""
import torch
import numpy as np
from lgdet.util.util_nms.util_nms_python import NMS


class ParsePredict_yolo:
    # TODO: 分解parseprdict.
    def __init__(self, cfg):
        self.cfg = cfg
        self.anchors = torch.Tensor(cfg.TRAIN.ANCHORS[::-1]) # SMALL -> BIG
        self.anc_num = cfg.TRAIN.FMAP_ANCHOR_NUM
        self.cls_num = cfg.TRAIN.CLASSES_NUM
        self.NMS = NMS(cfg)
        self.device = self.cfg.TRAIN.DEVICE
        self.boxlosstype, self.objlosstype = self.cfg.TRAIN.LOSSTYPE
        if 'yolov5' in self.cfg.TRAIN.MODEL:
            self.grid_sensitive = 2  # pp-yolo set 1.05    # Grid Sensitive [-1 or >1]
            self.scale_wh = True
        else:
            self.grid_sensitive = -1  # 1.05 #    # Grid Sensitive [-1 or >1]
            self.scale_wh = False
        self.grid = [torch.zeros(1)] * self.anc_num  # init grid
        self.iou_aware_factor = 0.4

    def parse_predict(self, f_maps):
        """
        Parse the predict. with all feature maps to labels.

        :param f_maps: predictions out of net.
        :return:parsed predictions, to labels.
        """
        pre_obj = []
        pre_cls = []
        pre_loc = []
        for f_id, f_map in enumerate(f_maps):
            # Parse one feature map.
            _pre_obj, _pre_cls, _pre_relative_box = self._parse_yolo_predict_fmap(f_map, f_id=f_id, tolabel=True)
            _pre_obj = _pre_obj.unsqueeze(-1)
            BN = _pre_obj.shape[0]
            pre_obj.append(_pre_obj.reshape(BN, -1, _pre_obj.shape[-1]))
            pre_cls.append(_pre_cls.reshape(BN, -1, _pre_cls.shape[-1]))
            pre_loc.append(_pre_relative_box.reshape(BN, -1, _pre_relative_box.shape[-1]))

        pre_obj = torch.cat(pre_obj, -2)
        pre_cls = torch.cat(pre_cls, -2)
        pre_loc = torch.cat(pre_loc, -2)

        labels_predict = []
        BN = pre_obj.shape[0]
        pre_obj_thresh = 0
        if pre_obj_thresh:
            for bi in range(BN):
                # score is conf*cls,but obj is pre_obj>thresh.
                mask = pre_obj[bi] > self.cfg.TEST.SCORE_THRESH
                if torch.sum(mask) > 0:
                    pre_obj_i = pre_obj[bi][mask]
                    pre_cls_i = pre_cls[bi][mask.squeeze()]
                    pre_cls_score_i, pre_cls_id = pre_cls_i.max(-1)
                    pre_score = pre_obj_i * pre_cls_score_i  # score of obj * score of class.
                    pre_loc_i = pre_loc[bi][mask.squeeze()]
                    labels_predict.append(self.NMS._nms(pre_score, pre_cls_id, pre_loc_i, xywh2x1y1x2y2=True))
                else:
                    labels_predict.append([])
        else:
            #  conf*cls>thresh

            pre_score = pre_obj * pre_cls
            labels_predict = self.NMS.forward(pre_score, pre_loc)

        return labels_predict

    def _parse_yolo_predict_fmap(self, f_map, f_id, tolabel=False):
        # pylint: disable=no-self-use
        """
        Reshape the predict, or reshape it to label shape.

        :param predict: out of net
        :param tolabel: True or False to label shape
        :return:
        """
        # make the feature map anchors

        # 1: pre deal feature mps
        B, C, H, W = f_map.shape

        anchor_ch, grid_wh = self._make_anc(f_id, W, H)
        if self.grid[f_id].shape[2:4] != f_map.shape[2:4]:
            self.grid[f_id] = self._make_grid(W, H).float().to(self.device)

        base = 6 if self.cfg.TRAIN.IOU_AWARE else 5
        f_map = f_map.view(B, self.anc_num, self.cls_num + base, H, W)
        permiute_type = (0, 1, 3, 4, 2)
        f_map = f_map.permute(permiute_type).contiguous()

        pred_conf = SigmodNot(f_map[..., 0])
        pred_xy = SigmodNot(f_map[..., 1:3])
        pred_wh = SigmodNot(f_map[..., 3:5])
        pred_cls = SigmodNot(f_map[..., base:])
        pred_iou = f_map[..., base - 1] if self.cfg.TRAIN.IOU_AWARE else None

        pred_conf.sigmoid()

        # Grid Sensitive
        if self.grid_sensitive > 1:
            pred_xy.sigmoid()
            pred_xy.value = self.grid_sensitive * pred_xy.value - 0.5 * (self.grid_sensitive - 1.0)  # Grid Sensitive
        if self.scale_wh:
            pred_wh.sigmoid()
            pred_wh.value = (pred_wh.value * 2) ** 2  # Width
            _pre_wh = pred_wh.value * anchor_ch
        else:
            assert pred_wh.sigmoid_tag == False
            _pre_wh = pred_wh.value.exp() * anchor_ch

        _pre_xy = pred_xy.value + self.grid[f_id]
        pre_box = torch.cat([_pre_xy, _pre_wh], -1) / grid_wh

        if not tolabel:  # training
            return pred_conf, pred_cls, pred_xy, pred_wh, pred_iou, pre_box
        else:  # test
            if self.cfg.TRAIN.IOU_AWARE:
                new_obj = torch.pow(pred_conf.value, (1 - self.iou_aware_factor)) * torch.pow(pred_iou, self.iou_aware_factor)
                eps = 1e-7
                new_obj = torch.clamp(new_obj, eps, 1 / eps)
                one = torch.ones_like(new_obj)
                new_obj = torch.clamp((one / new_obj - 1.0), eps, 1 / eps)
                pred_conf = (-torch.log(new_obj)).sigmoid()
            if pred_conf.sigmoid_tag is False:
                pred_conf.sigmoid()
            if pred_cls.sigmoid_tag is False:
                pred_cls.sigmoid()
            return pred_conf.value, pred_cls.value, pre_box

    def _make_grid(self, nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def _make_anc(self, f_id, W, H):
        mask = np.arange(self.anc_num) + self.anc_num * f_id
        anchors_raw = self.anchors[mask]
        anchors = torch.Tensor(
            [(a_w / self.cfg.TRAIN.IMG_SIZE[1] * W, a_h / self.cfg.TRAIN.IMG_SIZE[0] * H) for a_w, a_h in
             anchors_raw]).to(self.device)
        grid_wh = torch.Tensor([W, H, W, H]).to(self.device)
        anchor_ch = anchors.view(1, self.anc_num, 1, 1, 2)
        return anchor_ch, grid_wh

    def _parse_yolo_predict_fmap_old(self, f_map, f_id, tolabel=False):
        # pylint: disable=no-self-use
        """
        Reshape the predict, or reshape it to label shape.

        :param predict: out of net
        :param tolabel: True or False to label shape
        :return:
        """
        # make the feature map anchors
        mask = np.arange(self.anc_num) + self.anc_num * f_id
        anchors = self.anchors[mask]
        f_map = f_map.permute(0, 2, 3, 1)
        ###1: pre deal feature mps
        obj_pred, cls_perd, loc_pred = torch.split(f_map, [self.anc_num, self.anc_num * self.cls_num, self.anc_num * 4], 3)
        pre_obj = obj_pred.sigmoid()
        cls_reshape = torch.reshape(cls_perd, (-1, self.cls_num))
        cls_pred_prob = torch.softmax(cls_reshape, -1)
        pre_cls = torch.reshape(cls_pred_prob, cls_perd.shape)
        pre_loc = loc_pred

        batch_size = pre_obj.shape[0]
        shape = pre_obj.shape[1:3]

        pre_obj = pre_obj.unsqueeze(-1)
        pre_cls = pre_cls.reshape([batch_size, shape[0], shape[1], self.anc_num, self.cls_num])

        # reshape the pre_loc
        pre_loc = pre_loc.reshape([batch_size, shape[0], shape[1], self.anc_num, 4])
        pre_loc_xy = pre_loc[..., 0:2].sigmoid()
        pre_loc_wh = pre_loc[..., 2:4]

        grid_x = torch.arange(0, shape[1]).view(-1, 1).repeat(1, shape[0]).unsqueeze(2).permute(1, 0, 2)
        grid_y = torch.arange(0, shape[0]).view(-1, 1).repeat(1, shape[1]).unsqueeze(2)
        grid_xy = torch.cat([grid_x, grid_y], 2).unsqueeze(2).unsqueeze(0). \
            expand(1, shape[0], shape[1], self.anc_num, 2).expand_as(pre_loc_xy).type(torch.FloatTensor).to(
            loc_pred.device)

        # prepare gird xy
        box_ch = torch.Tensor([shape[1], shape[0]]).to(loc_pred.device)
        pre_realtive_xy = (pre_loc_xy + grid_xy) / box_ch
        anchor_ch = anchors.view(1, 1, 1, self.anc_num, 2).expand(1, shape[0], shape[1], self.anc_num, 2).to(
            loc_pred.device)
        pre_realtive_wh = pre_loc_wh.exp() * anchor_ch

        pre_relative_box = torch.cat([pre_realtive_xy, pre_realtive_wh], -1)

        if tolabel:
            return pre_obj, pre_cls, pre_relative_box
        return pre_obj, pre_cls, pre_relative_box, pre_loc_xy, pre_loc_wh, grid_xy, shape


class SigmodNot:
    def __init__(self, value):
        self.sigmoid_tag = False
        self.value = value
        self.value_raw = value
        self.infos = ''

    def sigmoid(self):
        self.value = self.value.sigmoid()
        self.sigmoid_tag = True
