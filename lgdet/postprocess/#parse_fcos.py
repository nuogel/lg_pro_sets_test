"""Parse the predictions."""
import torch
from lgdet.util.util_nms.util_nms_python import NMS
from lgdet.util.util_lg_transformer import LgTransformer


class ParsePredict_fcos:
    # TODO: 分解parseprdict.
    def __init__(self, cfg):
        self.cfg = cfg
        self.anchors = torch.Tensor(cfg.TRAIN.ANCHORS)
        self.anc_num = cfg.TRAIN.FMAP_ANCHOR_NUM
        self.cls_num = cfg.TRAIN.CLASSES_NUM
        self.NMS = NMS(cfg)
        self.device = self.cfg.TRAIN.DEVICE
        self.transformer = LgTransformer(self.cfg)

    def parse_predict(self, predicts, softmax=False):
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

        labels_predict = self._parse_multi_boxes(scores, locs)

        return labels_predict
