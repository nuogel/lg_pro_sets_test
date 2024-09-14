from lgdet.util.util_nms.util_nms_python import NMS
import torch

def NMS_block(predicts, cfg):
    if predicts == [[]]:
        return predicts
    nms = NMS(cfg)
    labels_out=[]
    for predict in predicts:
        pre_score, pre_class, pre_loc = [], [], []
        for lables in predict:
            pre_score.append(lables[0])
            pre_class.append(lables[1])
            pre_loc.append(lables[2])
        pre_score = torch.Tensor(pre_score)
        pre_class = torch.Tensor(pre_class)
        pre_loc = torch.Tensor(pre_loc)
        keep = nms.NMS_Greedy(pre_score, pre_class, pre_loc)

        labels_out.append(nms.nms2labels(keep, pre_score, pre_class, pre_loc, xywh2x1y1x2y2=False))
    return labels_out
