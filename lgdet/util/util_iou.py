"""IOU calculation between box1 and box2, box can be any shape."""
import torch
import math
import numpy as np


def _iou_mat_N2N_yolo(box1, box2):
    """
    IOU calculation between box1 and box2,in the shape of [x1 y1 x2 y2].

    :param box1:in the shape of [x1 y1 x2 y2]  #[-0.000, 0.6662, 0.0773...
    :param box2:in the shape of [x1 y1 x2 y2]
    :return:IOU of boxes1, boxes2
    """
    if len(box1.shape) == 1:
        box1 = box1.view(1, box1.shape[0])
    if len(box2.shape) == 1:
        box2 = box2.view(1, box2.shape[0])
    ious = torch.zeros((box2.shape[0],) + box1.shape[:-1]).to(box1.device)
    for i in range(box2.shape[0]):
        box2_ = box2[i]
        box_tmp = {}
        area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
        area2 = (box2_[..., 2] - box2_[..., 0]) * (box2_[..., 3] - box2_[..., 1])
        #
        box_tmp['x1'] = torch.max(box1[..., 0], box2_[..., 0])
        box_tmp['y1'] = torch.max(box1[..., 1], box2_[..., 1])
        box_tmp['x2'] = torch.min(box1[..., 2], box2_[..., 2])
        box_tmp['y2'] = torch.min(box1[..., 3], box2_[..., 3])
        box_w = (box_tmp['x2'] - box_tmp['x1'])
        box_h = (box_tmp['y2'] - box_tmp['y1'])
        intersection = box_w * box_h
        ious[i] = intersection / (area1 + area2 - intersection)
        #
        mask1 = (box1[..., 0] + box1[..., 2] - box2_[..., 2] - box2_[..., 0]).abs() \
                - (box1[..., 2] - box1[..., 0] + box2_[..., 2] - box2_[..., 0])
        mask1 = (mask1 < 0).float().to(box1.device)
        mask2 = (box1[..., 1] + box1[..., 3] - box2_[..., 3] - box2_[..., 1]).abs() \
                - (box1[..., 3] - box1[..., 1] + box2_[..., 3] - box2_[..., 1])
        mask2 = (mask2 < 0).float().to(box1.device)
        #
        ious[i] = ious[i] * mask1 * mask2
    return ious


def _iou_mat_N2N(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU


def _iou_mat_N21(box1, box2):
    '''
    如果shape[N1, 4] iou [N2, 4]，则 N1=N2,一般用法为[N,4], [1, 4] 输出为[N], 如果
    IOU calculation between box1 and box2,in the shape of [x1 y1 x2 y2].

    :param box1:in the shape of [x1 y1 x2 y2]  #[-0.000, 0.6662, 0.0773...
    :param box2:in the shape of [x1 y1 x2 y2]
    :return:IOU of boxes1, boxes2
    '''
    area1 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    area2 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    iw = torch.min(box1[..., 2], box2[..., 2]) - torch.max(box1[..., 0], box2[..., 0])
    ih = torch.min(box1[..., 3], box2[..., 3]) - torch.max(box1[..., 1], box2[..., 1])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    intersection = iw * ih
    ua = area1 + area2 - intersection
    IoU = intersection / ua + 1e-16
    return IoU


def iou_xywh(boxes1, boxes2, type='N2N'):
    """
    IOU calculation between box1 and box2,which is in the shape of [x y w h].

    :param boxes1:in the shape of [x y w h]
    :param boxes2:in the shape of [x y w h]
    :return:IOU of boxes1, boxes2
    """
    boxes1 = xywh2xyxy(boxes1)
    boxes2 = xywh2xyxy(boxes2)
    ious = _iou_mat(boxes1, boxes2, type)
    return ious


def iou_xyxy(boxes1, boxes2, type='N2N'):
    """
    IOU calculation between box1 and box2,which is in the shape of [x y x y]

    :param boxes1:in the shape of [x y x y]
    :param boxes2:in the shape of [x y x y]
    :return:IOU of boxes1, boxes2
    """
    ious = _iou_mat(boxes1, boxes2, type)
    return ious


def _iou_mat(box1, box2, type='N2N'):
    '''

    :param boxes1:in the shape of [x y x y]
    :param boxes2:in the shape of [x y x y]
    :param type:
    :return:
    '''

    if type == 'N2N':
        return _iou_mat_N2N(box1, box2)
    elif type == 'N2N_yolo':
        return _iou_mat_N2N_yolo(box1, box2)
    else:
        return _iou_mat_N21(box1, box2)


def xywh2xyxy(boxes_xywh):
    """
    Convert boxes with the shape xywh to x1y1x2y2.
    """

    boxes_xyxy = torch.cat([boxes_xywh[..., :2] - boxes_xywh[..., 2:] / 2.0,
                            boxes_xywh[..., :2] + boxes_xywh[..., 2:] / 2.0], -1)
    return boxes_xyxy


def xyxy2xywh(boxes_xyxy):
    """
    Convert boxes with the shape xywh to x1y1x2y2.
    """

    boxes_xywh = torch.cat(
        [(boxes_xyxy[..., :2] + boxes_xyxy[..., 2:]) / 2.0, boxes_xyxy[..., 2:] - boxes_xyxy[..., :2]], -1)
    return boxes_xywh


# def _iou_wh(wh1, wh2):
#     '''
#     only [w,h]
#     :param wh1:
#     :param wh2:
#     :return:
#     '''
#     wh2 = wh2.t()
#     w1, h1 = wh1[0], wh1[1]
#     w2, h2 = wh2[0], wh2[1]
#     inter_area = torch.min(w1, w2) * torch.min(h1, h2)
#     union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
#     return inter_area / union_area


def _iou_wh(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


def bbox_GDCiou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

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
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


def _bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def _iou_np(cubes_a, cubes_b):
    """
    numpy 计算IoU
    :param cubes_a: [N,(x1,y1,x2,y2)]
    :param cubes_b: [M,(x1,y1,x2,y2)]
    :return:  IoU [N,M]
    """
    # expands dim
    cubes_a = np.expand_dims(cubes_a, axis=1)  # [N,1,4]
    cubes_b = np.expand_dims(cubes_b, axis=0)  # [1,M,4]
    overlap = np.maximum(0.0,
                         np.minimum(cubes_a[..., 2:], cubes_b[..., 2:]) -
                         np.maximum(cubes_a[..., :2], cubes_b[..., :2]))  # [N,M,(w,h)]

    # overlap
    overlap = np.prod(overlap, axis=-1)  # [N,M]

    # compute area
    area_a = np.prod(cubes_a[..., 2:] - cubes_a[..., :2], axis=-1)
    area_b = np.prod(cubes_b[..., 2:] - cubes_b[..., :2], axis=-1)

    # compute iou
    iou = overlap / (area_a + area_b - overlap)
    return iou


def box_iou(boxes1, boxes2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(boxes1.t())
    area2 = box_area(boxes2.t())

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    inter = (rb - lt).clamp(min=0).prod(2)  # [N,M]
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)
