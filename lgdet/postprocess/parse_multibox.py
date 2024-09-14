"""Parse the predictions."""
import torch
from lgdet.util.util_nms.util_nms_python import NMS
from lgdet.util.util_lg_transformer import LgTransformer


class ParsePredict_multibox:
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
        pre_score, pre_loc, anchors_xywh = predicts
        if softmax:
            pre_score = pre_score.softmax(-1)  # conf preds
        pre_loc_xywh = self._decode_bboxes(pre_loc, anchors_xywh)
        labels_predict = self.NMS.forward(pre_score, pre_loc_xywh)
        return labels_predict

    def _decode_bboxes(self, locations, priors):
        bboxes = torch.cat([
            locations[..., :2] * 0.1 * priors[..., 2:] + priors[..., :2],
            torch.exp(locations[..., 2:] * 0.2) * priors[..., 2:]
        ], dim=locations.dim() - 1)
        return bboxes
