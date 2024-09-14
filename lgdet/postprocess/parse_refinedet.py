from lgdet.loss.loss_refinedet import Detect_RefineDet
from lgdet.util.util_iou import xyxy2xywh
from lgdet.util.util_nms.util_nms_python import NMS
import torch


class ParsePredict_refinedet:
    # TODO: 分解parseprdict.
    def __init__(self, cfg):
        self.cfg = cfg
        self.anchors = torch.Tensor(cfg.TRAIN.ANCHORS)
        self.anc_num = cfg.TRAIN.FMAP_ANCHOR_NUM
        self.cls_num = cfg.TRAIN.CLASSES_NUM
        self.NMS = NMS(cfg)
        self.device = self.cfg.TRAIN.DEVICE

    def parse_predict(self, predicts):
        detecte = Detect_RefineDet(self.cfg)
        pre_score, pre_loc = detecte.forward(predicts)
        pre_loc_xywh = xyxy2xywh(pre_loc)
        labels_predict = self.NMS.forward(pre_score, pre_loc_xywh)
        return labels_predict
