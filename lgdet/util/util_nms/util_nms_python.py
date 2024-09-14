import torch
import numpy as np
from lgdet.util.util_iou import iou_xywh, iou_xyxy, xywh2xyxy as _xywh2xyxy
import torchvision
from lgdet.util.util_iou import box_iou

try:
    from lgdet.util.util_nms.nms_wrapper import nms as _nms_cython

    nms_cython_ok = 1
except:
    nms_cython_ok = 0


class NMS:  # TODO: dubug the for ...in each NMS.
    def __init__(self, cfg):
        self.cfg = cfg
        self.score_thresh = cfg.TEST.SCORE_THRESH
        self.theta = cfg.TEST.SOFNMS_THETA
        self.iou_thresh = cfg.TEST.IOU_THRESH
        self.class_range = range(cfg.TRAIN.CLASSES_NUM)
        if nms_cython_ok and self.cfg.TEST.USE_MNS_CYTHON:
            self.use_nms_cython = True
            print('use nms-cython')
        else:
            self.use_nms_cython = False
            print('use nms-lg')
        self.max_detection_boxes_num = 300
        if self.cfg.TEST.MAP_FSCORE in [0, '0']:  # 1-fscore
            self.score_thresh = 0.05  # count map score thresh is <0.05
            self.max_detection_boxes_num = 300

    def forward(self, pre_score, pre_loc, xywh2xyxy=True):
        if self.use_nms_cython:
            return self.nms_cython(pre_score, pre_loc, xywh2xyxy)
        else:
            return self.nms_lg(pre_score, pre_loc, xywh2xyxy)

    def nms_lg(self, pre_score, pre_loc, xywh2xyxy=True):
        labels_predict = []
        for batch_n in range(pre_score.shape[0]):
            pre_score_max = pre_score[batch_n].max(-1)
            _pre_score = pre_score_max[0]
            _pre_class = pre_score_max[1]
            _pre_loc = pre_loc[batch_n]

            index = _pre_score > self.score_thresh
            _pre_score = _pre_score[index]
            _pre_class = _pre_class[index]
            _pre_loc = _pre_loc[index]

            # score_sort = _pre_score.sort(descending=True)
            # score_idx = score_sort[1][:10000]
            # _pre_score = _pre_score[score_idx]
            # _pre_class = _pre_class[score_idx]
            # _pre_loc = _pre_loc[score_idx]

            # TODO: simple code
            # _pre_score = pre_score[batch_n]
            # index = _pre_score > self.score_thresh
            # _pre_score = _pre_score[index]
            # _pre_loc = pre_loc[batch_n][index]

            labels = self._nms(_pre_score, _pre_class, _pre_loc, xywh2xyxy)
            labels_predict.append(labels)

        return labels_predict

    def nms_cython(self, pre_score, pre_loc, xywh2xyxy=True):
        labels_predict = []
        for batch_n in range(pre_score.shape[0]):
            pred_i = []
            scores = pre_score[batch_n]
            boxes = pre_loc[batch_n]
            if xywh2xyxy:
                boxes = _xywh2xyxy(boxes)
            for j in self.class_range:
                inds = (scores[:, j] > self.score_thresh)
                if inds.sum() == 0:
                    continue
                c_bboxes = boxes[inds].cpu().detach().numpy()
                c_scores = scores[inds, j].cpu().detach().numpy()
                c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(np.float32, copy=False)
                cpu = False
                keep = _nms_cython(c_dets, self.iou_thresh, force_cpu=cpu)
                keep = keep[:self.max_detection_boxes_num]
                c_dets = c_dets[keep, :]
                c_dets_reshap = c_dets.copy()
                c_dets_reshap[..., 0] = c_dets[..., -1]
                c_dets_reshap[..., 1:] = c_dets[..., :4]
                c_dets_reshap = np.insert(c_dets_reshap, 1, j, 1)
                pred_i.append(c_dets_reshap)
            pred_i = np.concatenate(pred_i)
            labels_predict.append(pred_i)
        return labels_predict

    def nms2labels(self, keep, pre_score, pre_class, pre_loc, xywh2xyxy):
        # labels_out = []
        # pre_loc = pre_loc.cpu().detach().numpy()
        # for keep_idx in keep:
        #     box = pre_loc[keep_idx]
        #     box = box.squeeze()
        #     if xywh2xyxy:
        #         boxx1 = max(0, box[0] - box[2] / 2)
        #         boxy1 = max(0, box[1] - box[3] / 2)
        #         boxx2 = max(0, box[0] + box[2] / 2)
        #         boxy2 = max(0, box[1] + box[3] / 2)
        #     else:
        #         boxx1 = max(0, box[0])
        #         boxy1 = max(0, box[1])
        #         boxx2 = max(0, box[2])
        #         boxy2 = max(0, box[3])
        #     pre_score_out = pre_score[keep_idx].item()
        #     class_out = pre_class[keep_idx].item()
        #     labels_out.append([pre_score_out, class_out, boxx1, boxy1, boxx2, boxy2])

        pre_loc = xywh2xyxy(pre_loc)
        labels_out = torch.cat((pre_score.unsqueeze(-1), pre_class.unsqueeze(-1), pre_loc), -1)
        labels_out = labels_out[keep]
        return labels_out

    def _nms(self, pre_score=None, pre_class=None, pre_loc=None, xywh2xyxy=None):
        if self.cfg.TEST.NMS_TYPE in ['soft_nms', 'SOFT_NMS']:
            keep = self.NMS_Soft(pre_score, pre_class, pre_loc)
        elif self.cfg.TEST.NMS_TYPE in ['greedy']:
            keep = self.NMS_Greedy(pre_score, pre_class, pre_loc)
        else:
            keep = self.NMS_torchvision(pre_score, pre_class, pre_loc)

        keep = keep[:self.max_detection_boxes_num]
        if xywh2xyxy:
            pre_loc = _xywh2xyxy(pre_loc)
        labels_out = torch.cat((pre_score.unsqueeze(-1), pre_class.unsqueeze(-1), pre_loc), -1)
        labels_out = labels_out[keep]

        # labels_out = self.nms2labels(keep, pre_score, pre_class, pre_loc, xywh2xyxy)

        return labels_out

    def NMS_Greedy(self, pre_score, pre_class, pre_loc):
        """
        Nms.

        :param pre_score: in the shape of [box_number, class_number]
        :param pre_loc: int the shape of [box_number, 4] 4 means [x1, y1, x2, y2]
        :param score_thresh:score_thresh
        :param iou_thresh:iou_thresh
        :return: labels_out
        """
        # print('using Greedy NMS')

        score_sort = pre_score.sort(descending=True)  # sort the scores.
        score_idx = score_sort[1]

        keep = []
        for i in self.class_range:  # with different classess.
            a = pre_class[score_idx] == i  # each class for NMS.
            order_index = score_idx[a]  # get the index of orders
            while order_index.shape[0] > 0:  # deal with all the boxes.
                max_one = order_index[0].item()  # get index of the max score box.
                box_head = pre_loc[max_one]  # get the score of it
                box_others = pre_loc[order_index[1:]]  # the rest boxes.
                ious = iou_xywh(box_head, box_others, type='N21')  # count the ious between the max one and the others
                rest = torch.lt(ious, self.iou_thresh).squeeze()  # find the boxes of iou<0.5(thresh), discard the iou>0.5.
                order_index = order_index[1:][rest]  # get the new index of the rest boxes, except the max one.
                keep.append(max_one)
        return keep

    def NMS_Soft(self, pre_score, pre_class, pre_loc):
        """
           Nms.

           :param pre_score: in the shape of [box_number, class_number]
           :param pre_loc: int the shape of [box_number, 4] 4 means [x, y, w, h]
           :param score_thresh:score_thresh
           :param iou_thresh:iou_thresh
           :return: labels_out
           """
        # print('using Soft NMS')

        score_sort = pre_score.sort(descending=True)
        score_idx = score_sort[1]

        keep = []
        for i in self.class_range:  # with different classess.
            a = pre_class[score_idx] == i
            order_index = score_idx[a]
            while order_index.shape[0] > 0:
                max_one = order_index[0].item()
                keep.append(max_one)
                box_head = pre_loc[max_one]
                box_others = pre_loc[order_index[1:]]
                score_others = pre_score[order_index[1:]]
                # print(score_others)
                ious = iou_xywh(box_head, box_others, type='N21').reshape(1, -1)
                # print('iou', ious)
                soft_score = score_others * torch.exp(-pow(ious, 2) / self.theta).squeeze()  # s = s*e^(-iou^2 / theta)
                # print('s',soft_score)
                rest = torch.gt(soft_score, self.iou_thresh)
                order_index = order_index[1:][rest]
                new_index = pre_score[order_index].sort(descending=True)
                order_index = order_index[new_index[1]]

        # ###########  NMS UP  ####################
        return keep

    def NMS_torchvision(self, pre_score, pre_class, pre_loc):  # use offset and very fast
        if pre_loc.numel() == 0:
            return []
        pre_loc = _xywh2xyxy(pre_loc)
        max_coordinate = pre_loc.max()
        offsets = pre_class.to(pre_loc) * (max_coordinate + 1)
        boxes_for_nms = pre_loc + offsets[:, None]
        keep = torchvision.ops.nms(boxes_for_nms, pre_score, self.iou_thresh)
        return keep

    def NMS_SSD(self, boxes, scores, overlap=0.5, top_k=200):
        """Apply non-maximum suppression at test time to avoid detecting too many
        overlapping bounding boxes for a given object.
        Args:
            boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
            scores: (tensor) The class predscores for the img, Shape:[num_priors].
            overlap: (float) The overlap thresh for suppressing unnecessary boxes.
            top_k: (int) The Maximum number of box preds to consider.
        Return:
            The indices of the kept boxes with respect to num_priors.
        """

        keep = scores.new(scores.size(0)).zero_().long()
        if boxes.numel() == 0:
            return keep
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        area = torch.mul(x2 - x1, y2 - y1)
        v, idx = scores.sort(0)  # sort in ascending order
        # I = I[v >= 0.01]
        idx = idx[-top_k:]  # indices of the top-k largest vals
        xx1 = boxes.new()
        yy1 = boxes.new()
        xx2 = boxes.new()
        yy2 = boxes.new()
        w = boxes.new()
        h = boxes.new()

        # keep = torch.Tensor()
        count = 0
        while idx.numel() > 0:
            i = idx[-1]  # index of current largest val
            # keep.append(i)
            keep[count] = i
            count += 1
            if idx.size(0) == 1:
                break
            idx = idx[:-1]  # remove kept element from view
            # load bboxes of next highest vals
            torch.index_select(x1, 0, idx, out=xx1)
            torch.index_select(y1, 0, idx, out=yy1)
            torch.index_select(x2, 0, idx, out=xx2)
            torch.index_select(y2, 0, idx, out=yy2)
            # store element-wise max with next highest score
            xx1 = torch.clamp(xx1, min=x1[i])
            yy1 = torch.clamp(yy1, min=y1[i])
            xx2 = torch.clamp(xx2, max=x2[i])
            yy2 = torch.clamp(yy2, max=y2[i])
            w.resize_as_(xx2)
            h.resize_as_(yy2)
            w = xx2 - xx1
            h = yy2 - yy1
            # check sizes of xx1 and xx2.. after each iteration
            w = torch.clamp(w, min=0.0)
            h = torch.clamp(h, min=0.0)
            inter = w * h
            # IoU = i / (area(a) + area(b) - i)
            rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
            union = (rem_areas - inter) + area[i]
            IoU = inter / union  # store result in iou
            # keep only elements with an IoU <= overlap
            idx = idx[IoU.le(overlap)]
        return keep, count

    def NMS_from_FAST_RCNN(self, pre_score_raw, pre_loc, score_thresh, iou_thresh):  # NMS changed from FAST RCNN
        """
        Nms.

        :param pre_score: in the shape of [box_number, class_number]
        :param pre_loc: int the shape of [box_number, 4] 4 means [x1, y1, x2, y2]
        :param score_thresh:score_thresh
        :param iou_thresh:iou_thresh
        :return: labels_out
        """
        # print('using NMS_from_FAST_RCNN')
        pre_score_raw = pre_score_raw.max(-1)
        scores = pre_score_raw[0]
        pre_class = pre_score_raw[1]

        x1 = pre_loc[:, 0]
        y1 = pre_loc[:, 1]
        x2 = pre_loc[:, 2]
        y2 = pre_loc[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.sort(descending=True)[1]
        keep = []
        while order.shape[0] > 0:
            i = order[0]
            keep.append(i)
            xx1 = torch.max(x1[i], x1[order[1:]])
            yy1 = torch.max(y1[i], y1[order[1:]])
            xx2 = torch.min(x2[i], x2[order[1:]])
            yy2 = torch.min(y2[i], y2[order[1:]])
            w = torch.max(0.0, xx2 - xx1 + 1)
            h = torch.max(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_thresh)[0]
            order = order[inds + 1]
        return keep

    def NMS_old_edition(self, pre_score_raw, pre_loc, score_thresh, iou_thresh):
        """
        Nms.

        :param pre_score: in the shape of [box_number, class_number]
        :param pre_loc: int the shape of [box_number, 4] 4 means [x1, y1, x2, y2]
        :param score_thresh:score_thresh
        :param iou_thresh:iou_thresh
        :return: labels_out
        """
        pre_score_raw = pre_score_raw.max(-1)
        pre_score = pre_score_raw[0]
        pre_class = pre_score_raw[1]
        idx = pre_score > score_thresh  # Return the index of the max pre_score
        print('max score:', torch.max(pre_score).item())
        pre_score_max = pre_score[idx]
        # print(pre_score[torch.gt(pre_score, pre_score_thre)])
        print('Num of box:', len(pre_score_max))
        if len(pre_score_max) > 1:
            sor = pre_score_max.sort(descending=True)[1]
            _idx = idx.nonzero()[sor]
            _leng = len(_idx)

            for i in range(_leng):
                for j in range(i + 1, _leng):
                    if pre_class[_idx[i]] != pre_class[_idx[j]]:  # diffent classes
                        continue
                    if pre_score[_idx[i]] < score_thresh or \
                            pre_score[_idx[j]] < score_thresh:
                        continue  # get out of the likely anchors which have been counted
                    box1 = pre_loc[_idx[i]].unsqueeze(0)
                    box2 = pre_loc[_idx[j]].unsqueeze(0)
                    iou_ = iou_xywh(box1, box2)
                    if iou_ > iou_thresh:
                        pre_score[_idx[j]] = 0.0
        _idx = (pre_score > score_thresh).nonzero()  # Return the index of the max pre_score
        labels_out = []
        for keep_idx in _idx:
            box = pre_loc[keep_idx]
            box = box.squeeze()
            boxx1 = box[0] - box[2] / 2
            boxy1 = box[1] - box[3] / 2
            boxx2 = box[0] + box[2] / 2
            boxy2 = box[1] + box[3] / 2
            box_out = [boxx1, boxy1, boxx2, boxy2]
            if min(box_out) <= 0:
                print('error!')
                continue
            pre_score_out = pre_score[keep_idx].item()
            class_out = pre_class[keep_idx].item()
            labels_out.append([pre_score_out, class_out, box_out])
        return labels_out

    # others:
    def fast_nms(self, boxes, scores, iou_thresh: float = 0.5):
        '''
        Arguments:
            boxes (Tensor[N, 4])
            scores (Tensor[N, 1])
        Returns:
            Fast NMS results
        '''
        scores, idx = scores.sort(1, descending=True)
        boxes = boxes[idx]  # 对框按得分降序排列
        iou = box_iou(boxes, boxes)  # IoU矩阵
        iou.triu_(diagonal=1)  # 上三角化
        keep = iou.max(dim=0)[0] < iou_thresh  # 列最大值向量，二值化

        return boxes[keep], scores[keep]

    def cluster_nms(self, boxes, scores, iou_thresh: float = 0.5):
        '''
        Arguments:
            boxes (Tensor[N, 4])
            scores (Tensor[N, 1])
        Returns:
            Fast NMS results
        '''
        scores, idx = scores.sort(1, descending=True)
        boxes = boxes[idx]  # 对框按得分降序排列
        iou = box_iou(boxes, boxes).triu_(diagonal=1)  # IoU矩阵，上三角化
        C = iou
        for i in range(200):
            A = C
            maxA = A.max(dim=0)[0]  # 列最大值向量
            E = (maxA < iou_thresh).float().unsqueeze(1).expand_as(A)  # 对角矩阵E的替代
            C = iou.mul(E)  # 按元素相乘
            if A.equal(C) == True:  # 终止条件
                break
        keep = maxA < iou_thresh  # 列最大值向量，二值化

        return boxes[keep], scores[keep]

    def SPM_cluster_nms(self, boxes, scores, iou_thresh: float = 0.5):
        '''
        Arguments:
            boxes (Tensor[N, 4])
            scores (Tensor[N, 1])
        Returns:
            Fast NMS results
        '''
        scores, idx = scores.sort(1, descending=True)
        boxes = boxes[idx]  # 对框按得分降序排列
        iou = box_iou(boxes, boxes).triu_(diagonal=1)  # IoU矩阵，上三角化
        C = iou
        for i in range(200):
            A = C
            maxA = A.max(dim=0)[0]  # 列最大值向量
            E = (maxA < iou_thresh).float().unsqueeze(1).expand_as(A)  # 对角矩阵E的替代
            C = iou.mul(E)  # 按元素相乘
            if A.equal(C) == True:  # 终止条件
                break
        scores = torch.prod(torch.exp(-C ** 2 / 0.2), 0) * scores  # 惩罚得分
        keep = scores > 0.01  # 得分阈值筛选
        return boxes[keep], scores[keep]
