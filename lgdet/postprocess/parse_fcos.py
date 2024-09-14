"""Parse the predictions."""
import torch.nn as nn
import torch
from lgdet.util.util_nms.util_nms_python import NMS
from lgdet.util.util_iou import xyxy2xywh
from lgdet.loss.loss_fcos import coords_fmap2orig


class DetectHead(nn.Module):
    def __init__(self, score_threshold, nms_iou_threshold, max_detection_boxes_num, strides, config=None):
        super().__init__()
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detection_boxes_num = max_detection_boxes_num
        self.strides = strides
        self.config = config
        self.device = self.config.TRAIN.DEVICE

    def forward(self, inputs):
        '''
        inputs  list [cls_logits,cnt_logits,reg_preds]
        cls_logits  list contains five [batch_size,class_num,h,w]
        cnt_logits  list contains five [batch_size,1,h,w]
        reg_preds   list contains five [batch_size,4,h,w]
        '''
        cls_logits, coords = self._reshape_cat_out(inputs[0], self.strides)  # [batch_size,sum(_h*_w),class_num]
        cnt_logits, _ = self._reshape_cat_out(inputs[1], self.strides)  # [batch_size,sum(_h*_w),1]
        reg_preds, _ = self._reshape_cat_out(inputs[2], self.strides)  # [batch_size,sum(_h*_w),4]

        cls_preds = cls_logits.sigmoid_()
        cnt_preds = cnt_logits.sigmoid_()

        coords = coords.to(self.device)

        # cls_scores, cls_classes = torch.max(cls_preds, dim=-1)  # [batch_size,sum(_h*_w)]
        if self.config.add_centerness:
            cls_preds = torch.sqrt(cls_preds * cnt_preds)  # [batch_size,sum(_h*_w)]
        # cls_classes = cls_classes + 1  # [batch_size,sum(_h*_w)]

        boxes_x1y1x2y2 = self._coords2boxes(coords, reg_preds)  # [batch_size,sum(_h*_w),4]

        return cls_preds, boxes_x1y1x2y2
        # # select topk
        # max_num = min(self.max_detection_boxes_num, cls_scores.shape[-1])
        # topk_ind = torch.topk(cls_scores, max_num, dim=-1, largest=True, sorted=True)[1]  # [batch_size,max_num]
        # _cls_scores = []
        # _cls_classes = []
        # _boxes = []
        # for batch in range(cls_scores.shape[0]):
        #     _cls_scores.append(cls_scores[batch][topk_ind[batch]])  # [max_num]
        #     _cls_classes.append(cls_classes[batch][topk_ind[batch]])  # [max_num]
        #     _boxes.append(boxes[batch][topk_ind[batch]])  # [max_num,4]
        # cls_scores_topk = torch.stack(_cls_scores, dim=0)  # [batch_size,max_num]
        # cls_classes_topk = torch.stack(_cls_classes, dim=0)  # [batch_size,max_num]
        # boxes_topk = torch.stack(_boxes, dim=0)  # [batch_size,max_num,4]
        # assert boxes_topk.shape[-1] == 4
        #
        # return self._post_process([cls_scores_topk, cls_classes_topk, boxes_topk])

    def _post_process(self, preds_topk):
        '''
        cls_scores_topk [batch_size,max_num]
        cls_classes_topk [batch_size,max_num]
        boxes_topk [batch_size,max_num,4]
        '''
        _cls_scores_post = []
        _cls_classes_post = []
        _boxes_post = []
        cls_scores_topk, cls_classes_topk, boxes_topk = preds_topk
        for batch in range(cls_classes_topk.shape[0]):
            mask = cls_scores_topk[batch] >= self.score_threshold
            _cls_scores_b = cls_scores_topk[batch][mask]  # [?]
            _cls_classes_b = cls_classes_topk[batch][mask]  # [?]
            _boxes_b = boxes_topk[batch][mask]  # [?,4]
            nms_ind = self.batched_nms(_boxes_b, _cls_scores_b, _cls_classes_b, self.nms_iou_threshold)
            _cls_scores_post.append(_cls_scores_b[nms_ind])
            _cls_classes_post.append(_cls_classes_b[nms_ind])
            _boxes_post.append(_boxes_b[nms_ind])
        # scores, classes, boxes = torch.stack(_cls_scores_post, dim=0), torch.stack(_cls_classes_post, dim=0), torch.stack(_boxes_post, dim=0)

        return _cls_scores_post, _cls_classes_post, _boxes_post

    @staticmethod
    def box_nms(boxes, scores, thr):
        '''
        boxes: [?,4]
        scores: [?]
        '''
        if boxes.shape[0] == 0:
            return torch.zeros(0, device=boxes.device).long()
        assert boxes.shape[-1] == 4
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.sort(0, descending=True)[1]
        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                i = order.item()
                keep.append(i)
                break
            else:
                i = order[0].item()
                keep.append(i)

            xmin = x1[order[1:]].clamp(min=float(x1[i]))
            ymin = y1[order[1:]].clamp(min=float(y1[i]))
            xmax = x2[order[1:]].clamp(max=float(x2[i]))
            ymax = y2[order[1:]].clamp(max=float(y2[i]))
            inter = (xmax - xmin).clamp(min=0) * (ymax - ymin).clamp(min=0)
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            idx = (iou <= thr).nonzero().squeeze()
            if idx.numel() == 0:
                break
            order = order[idx + 1]
        return torch.LongTensor(keep)

    def batched_nms(self, boxes, scores, idxs, iou_threshold):

        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)
        # strategy: in order to perform NMS independently per class.
        # we add an offset to all the boxes. The offset is dependent
        # only on the class idx, and is large enough so that boxes
        # from different classes do not overlap
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]
        keep = self.box_nms(boxes_for_nms, scores, iou_threshold)
        return keep

    def _coords2boxes(self, coords, offsets):
        '''
        Args
        coords [sum(_h*_w),2]
        offsets [batch_size,sum(_h*_w),4] ltrb
        '''
        x1y1 = coords[None, :, :] - offsets[..., :2]
        x2y2 = coords[None, :, :] + offsets[..., 2:]  # [batch_size,sum(_h*_w),2]
        boxes = torch.cat([x1y1, x2y2], dim=-1)  # [batch_size,sum(_h*_w),4]
        return boxes

    def _reshape_cat_out(self, inputs, strides):
        '''
        Args
        inputs: list contains five [batch_size,c,_h,_w]
        Returns
        out [batch_size,sum(_h*_w),c]
        coords [sum(_h*_w),2]
        '''
        batch_size = inputs[0].shape[0]
        c = inputs[0].shape[1]
        out = []
        coords = []
        for pred, stride in zip(inputs, strides):
            pred = pred.permute(0, 2, 3, 1)
            coord = coords_fmap2orig(pred, stride).to(device=pred.device)
            pred = torch.reshape(pred, [batch_size, -1, c])
            out.append(pred)
            coords.append(coord)
        return torch.cat(out, dim=1), torch.cat(coords, dim=0)


class ParsePredict_fcos:
    def __init__(self, config):
        self.detection_head = DetectHead(config.score_threshold, config.nms_iou_threshold,
                                         config.max_detection_boxes_num, config.strides, config)
        self.NMS = NMS(config)

    def parse_predict(self, predicts):
        pre_score, pre_loc_xyxy = self.detection_head(predicts)
        pre_loc_xywh = xyxy2xywh(pre_loc_xyxy)
        labels_predict = self.NMS.forward(pre_score, pre_loc_xywh)
        return labels_predict

    def reshape_predict(self, out):
        scores, classes, boxes = out
        reshape_out = []
        B = len(scores)
        for i in range(B):
            img_box = []
            for j, score in enumerate(scores[i]):
                img_box.append(
                    [scores[i][j].item(), classes[i][j].item(), boxes[i][j][0].item(), boxes[i][j][1].item(), boxes[i][j][2].item(), boxes[i][j][3].item()])
            reshape_out.append(img_box)
        return reshape_out

    def clip_boxes(self, size, batch_boxes):
        h, w = size
        batch_boxes = batch_boxes.clamp_(min=0)
        batch_boxes[..., [0, 2]] = batch_boxes[..., [0, 2]].clamp_(max=w - 1)
        batch_boxes[..., [1, 3]] = batch_boxes[..., [1, 3]].clamp_(max=h - 1)
        return batch_boxes
