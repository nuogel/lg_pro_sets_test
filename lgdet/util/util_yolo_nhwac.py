from __future__ import division
import torch


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area





def build_targets(pred_boxes, pred_cls, labels, anchors, ignore_thres):
    BoolTensor = torch.cuda.BoolTensor if pred_boxes.is_cuda else torch.BoolTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)
    nH = pred_boxes.size(1)
    nW = pred_boxes.size(2)
    nA = pred_boxes.size(3)
    nC = pred_cls.size(4)

    # Output tensors
    obj_mask = BoolTensor(nB, nH, nW, nA).fill_(0)
    noobj_mask = BoolTensor(nB, nH, nW, nA).fill_(1)
    class_mask = FloatTensor(nB, nH, nW, nA).fill_(0)
    iou_scores = FloatTensor(nB, nH, nW, nA).fill_(0)
    tx = FloatTensor(nB, nH, nW, nA).fill_(0)
    ty = FloatTensor(nB, nH, nW, nA).fill_(0)
    tw = FloatTensor(nB, nH, nW, nA).fill_(0)
    th = FloatTensor(nB, nH, nW, nA).fill_(0)
    tcls = FloatTensor(nB, nH, nW, nA, nC).fill_(0)

    target = labels.clone()
    # Convert to position relative to box
    target[:, 2::2] = labels[:, 2::2] * nW
    target[:, 3::2] = labels[:, 3::2] * nH

    target_boxes = target[:, 2:6]  # * nG
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]

    # Get anchors with best iou
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)  # tensor([0.4050, 0.6302, 0.4081, 0.3759, 0.0789, 0.4050, 0.6302, 0.4081, 0.3759, 0.0789], device='cuda:0')
    # Separate target values
    b, target_labels = target[:, :2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()
    # Set masks
    obj_mask[b, gj, gi, best_n] = 1
    noobj_mask[b, gj, gi, best_n] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], gj[i], gi[i], anchor_ious > ignore_thres] = 0

    # Coordinates
    tx[b, gj, gi, best_n] = gx - gx.floor()
    ty[b, gj, gi, best_n] = gy - gy.floor()
    # Width and height
    tw[b, gj, gi, best_n] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, gj, gi, best_n] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # One-hot encoding of label
    tcls[b, gj, gi, best_n, target_labels] = 1
    # Compute label correctness and iou at best anchor
    class_mask[b, gj, gi, best_n] = (pred_cls[b, gj, gi, best_n].argmax(-1) == target_labels).float()
    iou_scores[b, gj, gi, best_n] = bbox_iou(pred_boxes[b, gj, gi, best_n], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf


def to_cpu(tensor):
    return tensor.detach().cpu()
