"""Parse the predictions."""
import torch
import torch.nn as nn
import numpy as np
from util.util_NMS import NMS
from lgdet.loss.ObdLoss_REFINEDET import Detect_RefineDet
from util.util_iou import xywh2xyxy, xyxy2xywh


class ParsePredict:
    # TODO: 分解parseprdict.
    def __init__(self, cfg):
        self.cfg = cfg
        self.anchors = torch.Tensor(cfg.TRAIN.ANCHORS)
        self.anc_num = cfg.TRAIN.FMAP_ANCHOR_NUM
        self.cls_num = cfg.TRAIN.CLASSES_NUM
        self.NMS = NMS(cfg)
        self.device = self.cfg.TRAIN.DEVICE

    def _parse_predict(self, f_maps):
        PARSEDICT = {
            'yolov2': self._parse_yolo_predict,
            'yolov3': self._parse_yolo_predict,
            'yolov3_tiny': self._parse_yolo_predict,
            'yolov3_tiny_mobilenet': self._parse_yolo_predict,
            'yolov3_tiny_squeezenet': self._parse_yolo_predict,
            'yolov3_tiny_shufflenet': self._parse_yolo_predict,
            'yolov2_fgfa': self._parse_yolo_predict,
            'yolonano': self._parse_yolo_predict,
            'fcos': self._parse_fcos_predict,
            'refinedet': self._parse_refinedet_predict,
            'efficientdet': self._parse_ssd_predict,
            'ssdvgg': self._parse_ssd_predict,
        }

        labels_predict = PARSEDICT[self.cfg.TRAIN.MODEL](f_maps)
        return labels_predict

    def _predict2nms(self, pre_cls_score, pre_loc, xywh2x1y1x2y2=True):
        labels_predict = []
        for batch_n in range(pre_cls_score.shape[0]):
            # TODO: make a matrix instead of for...
            score = pre_cls_score[batch_n]
            loc = pre_loc[batch_n]
            labels = self.NMS.forward(score, loc, xywh2x1y1x2y2)
            labels_predict.append(labels)
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

        ###1: pre deal feature mps
        B, C, H, W = f_map.shape
        f_map = f_map.view(B, self.anc_num, self.cls_num + 5, H, W)
        f_map = f_map.permute(0, 3, 4, 1, 2).contiguous()  # ((0, 1, 3, 4, 2),   (0, 3, 4, 1, 2))

        # pred_x = torch.sigmoid(f_map[..., 0])  # Center x
        # pred_y = torch.sigmoid(f_map[..., 1])  # Center y
        # pred_w = f_map[..., 2]  # Width
        # pred_h = f_map[..., 3]  # Height
        # pred_conf = torch.sigmoid(f_map[..., 4])  # Conf
        # pred_cls = torch.sigmoid(f_map[..., 5:])  # Cls pred.

        # pred_xy = torch.sigmoid(f_map[..., 0:2])  # Center x
        # pred_wh = f_map[..., 2:4]  # Width
        # pred_conf = torch.sigmoid(f_map[..., 4])  # Conf
        # pred_cls = torch.sigmoid(f_map[..., 5:])  # Cls pred.

        pred_conf = torch.sigmoid(f_map[..., 0])  # Conf
        pred_xy = torch.sigmoid(f_map[..., 1:3])  # Center x
        pred_wh = f_map[..., 3:5]  # Width
        pred_cls = torch.sigmoid(f_map[..., 5:])  # Cls pred.
        if not tolabel:
            return pred_conf, pred_cls, pred_xy, pred_wh  ##pred_x, pred_y, pred_w, pred_h
        else:
            mask = np.arange(self.anc_num) + self.anc_num * f_id
            anchors_raw = self.anchors[mask]

            anchors = torch.Tensor([(a_w / self.cfg.TRAIN.IMG_SIZE[1] * W, a_h / self.cfg.TRAIN.IMG_SIZE[0] * H) for a_w, a_h in anchors_raw])
            anchor_ch = anchors.view(1, 1, 1, self.anc_num, 2).expand(1, H, W, self.anc_num, 2).to(self.device)
            grid_wh = torch.Tensor([W, H]).to(self.device)

            grid_x = torch.arange(W).repeat(H, 1).view([1, H, W, 1]).to(self.device)
            grid_y = torch.arange(H).repeat(W, 1).t().view([1, H, W, 1]).to(self.device)
            grid_xy = torch.cat([grid_x, grid_y], -1).unsqueeze(3).expand(1, H, W, self.anc_num, 2).to(self.device)

            pre_wh = pred_wh.exp() * anchor_ch
            pre_realtive_wh = pre_wh / grid_wh
            pre_realtive_xy = (pred_xy + grid_xy) / grid_wh

            pre_relative_box = torch.cat([pre_realtive_xy, pre_realtive_wh], -1)
            return pred_conf, pred_cls, pre_relative_box

    def _parse_yolo_predict(self, f_maps):
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
            _pre_obj = _pre_obj.reshape(BN, -1, _pre_obj.shape[-1])
            _pre_cls = _pre_cls.reshape(BN, -1, _pre_cls.shape[-1])
            _pre_loc = _pre_relative_box.reshape(BN, -1, _pre_relative_box.shape[-1])

            pre_obj.append(_pre_obj)
            pre_cls.append(_pre_cls)
            pre_loc.append(_pre_loc)

        pre_obj = torch.cat(pre_obj, -2)
        pre_cls = torch.cat(pre_cls, -2)
        pre_loc = torch.cat(pre_loc, -2)

        labels_predict = []
        BN = pre_obj.shape[0]
        for bi in range(BN):
            # score is conf*cls,but obj is pre_obj>thresh.
            mask = pre_obj[bi] > self.cfg.TEST.SCORE_THRESH
            pre_obj_i = pre_obj[bi][mask]
            pre_cls_score, pre_cls_id = pre_cls[bi].max(-1, keepdim=True)
            pre_cls_score = pre_cls_score[mask]
            pre_cls_id = pre_cls_id[mask]
            pre_score = pre_obj_i * pre_cls_score  # score of obj * score of class.
            pre_loc_i = pre_loc[bi][mask.squeeze(-1)]
            labels_predict.append(self.NMS.forward(pre_score, pre_cls_id, pre_loc_i, xywh2x1y1x2y2=True))
        return labels_predict

    def _parse_fcos_predict(self, predicts):

        def parse_one_feature(predict, feature_idx):
            stride = self.cfg.TRAIN.STRIDES[feature_idx]
            pre_cls, pre_ctness, pre_loc = predict
            pre_cls, pre_ctness, pre_loc = pre_cls.permute(0, 2, 3, 1), \
                                           pre_ctness.permute(0, 2, 3, 1), \
                                           pre_loc.permute(0, 2, 3, 1)
            # softmax the classes
            clsshape = pre_cls.shape
            N = clsshape[0]
            feature_size = [clsshape[1], clsshape[2]]

            pre_cls = pre_cls.sigmoid()
            # pre_cls = torch.reshape(pre_cls, (-1, 4)).softmax(-1)
            # pre_cls = torch.reshape(pre_cls, clsshape)

            pre_ctness = pre_ctness.sigmoid()
            if feature_idx == 2:
                i_0, i_1, i_2 = 0, 13, 13
                print('pre_loc[i_0, i_1, i_2]:', pre_loc[i_0, i_1, i_2])
                print('pre_cls[i_0, i_1, i_2]', pre_cls[i_0, i_1, i_2])
                print('pre_ctness[i_0, i_1, i_2]', pre_ctness[i_0, i_1, i_2])

            # print('pre_loc:', pre_loc[0, 13, 19])
            pre_loc = torch.exp(pre_loc)
            pre_cls_conf = pre_cls * pre_ctness

            # TODO:change the for ...to martrix
            # make a grid_xy
            pre_loc_xy = pre_loc[..., :2]
            grid_x = torch.arange(0, feature_size[1]).view(-1, 1).repeat(1, feature_size[0]) \
                .unsqueeze(2).permute(1, 0, 2)
            grid_y = torch.arange(0, feature_size[0]).view(-1, 1).repeat(1, feature_size[1]).unsqueeze(2)
            grid_xy = torch.cat([grid_x, grid_y], 2).unsqueeze(0).expand_as(pre_loc_xy).type(torch.cuda.FloatTensor)
            # print('grid_xy:', grid_xy[0, 1, 19])
            grid_xy = grid_xy * stride + stride / 2
            # print('grid_xy:', grid_xy[0, 1, 19])
            # print('pre_loc:', pre_loc_xy[0, 1, 19])
            x1 = grid_xy[..., 0] - pre_loc[..., 0]
            y1 = grid_xy[..., 1] - pre_loc[..., 1]
            w = pre_loc[..., 0] + pre_loc[..., 2]
            h = pre_loc[..., 1] + pre_loc[..., 3]
            x = x1 + w / 2
            y = y1 + h / 2
            if self.cfg.TRAIN.RELATIVE_LABELS:
                x /= self.cfg.TRAIN.IMG_SIZE[1]
                y /= self.cfg.TRAIN.IMG_SIZE[0]
                w /= self.cfg.TRAIN.IMG_SIZE[1]
                h /= self.cfg.TRAIN.IMG_SIZE[0]
            predicted_boxes = torch.stack([x, y, w, h], -1)

            return pre_cls_conf, predicted_boxes

        scores = []
        locs = []
        for feature_idx, predict in enumerate(predicts):
            pre_cls_conf, predicted_boxes = parse_one_feature(predict, feature_idx)
            if feature_idx == 2:
                i_0, i_1, i_2 = 0, 13, 13
                print('predicted_boxes[i_0, i_1, i_2]:', predicted_boxes[i_0, i_1, i_2])
                print('pre_cls_conf[i_0, i_1, i_2]', pre_cls_conf[i_0, i_1, i_2])

            batch_n = pre_cls_conf.shape[0]
            score = torch.reshape(pre_cls_conf, (batch_n, -1, pre_cls_conf.shape[-1]))
            loc = torch.reshape(predicted_boxes, (batch_n, -1, 4))
            scores.append(score)
            locs.append(loc)

        scores = torch.cat(scores, 1)
        locs = torch.cat(locs, 1)

        labels_predict = self._predict2nms(scores, locs)

        return labels_predict

    def _parse_ssd_predict(self, predicts):
        pre_score, loc, anchors_xywh = predicts
        pre_score = torch.softmax(pre_score, -1)  # conf preds
        pre_loc_xywh = self._decode_bboxes(loc, anchors_xywh)
        labels_predict = self._predict2nms(pre_score, pre_loc_xywh)
        return labels_predict

    def _parse_refinedet_predict(self, predicts):
        detecte = Detect_RefineDet(self.cfg)
        pre_score, pre_loc = detecte.forward(predicts)
        pre_loc_xywh = xyxy2xywh(pre_loc)
        labels_predict = self._predict2nms(pre_score, pre_loc_xywh)

        return labels_predict

    def _parse_efficientdet_predict(self, predicts):
        pre_score, pre_loc_xyxy = predicts
        pre_loc_xywh = xyxy2xywh(pre_loc_xyxy)
        labels_predict = self._predict2nms(pre_score, pre_loc_xywh)
        return labels_predict

    def _decode_bboxes(self, locations, priors):
        bboxes = torch.cat([
            locations[..., :2] * 0.1 * priors[..., 2:] + priors[..., :2],
            torch.exp(locations[..., 2:] * 0.2) * priors[..., 2:]
        ], dim=locations.dim() - 1)
        return bboxes


class DecodeBBox(nn.Module):

    def __init__(self, mean=None, std=None):
        super(DecodeBBox, self).__init__()
        if mean is None:
            self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32))
        else:
            self.mean = mean
        if std is None:
            self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32))
        else:
            self.std = std

    def forward(self, predicts, anchors):
        a_width = anchors[:, :, 2] - anchors[:, :, 0]
        a_height = anchors[:, :, 3] - anchors[:, :, 1]
        ctr_x = anchors[:, :, 0] + 0.5 * a_width
        ctr_y = anchors[:, :, 1] + 0.5 * a_height

        dx = predicts[:, :, 0] * self.std[0] + self.mean[0]
        dy = predicts[:, :, 1] * self.std[1] + self.mean[1]
        dw = predicts[:, :, 2] * self.std[2] + self.mean[2]
        dh = predicts[:, :, 3] * self.std[3] + self.mean[3]

        pred_ctr_x = ctr_x + dx * a_width
        pred_ctr_y = ctr_y + dy * a_height
        pred_w = torch.exp(dw) * a_width
        pred_h = torch.exp(dh) * a_height

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

        return pred_boxes
