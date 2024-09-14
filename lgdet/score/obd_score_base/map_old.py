"""Calculate the F1 score."""
import numpy as np
from lgdet.util.util_get_cls_names import _get_class_names
from lgdet.util.util_iou import _iou_np
from lgdet.registry import SCORES
from lgdet.postprocess.parse_factory import ParsePredict


@SCORES.registry()
class MAP:
    """Calculate the F1 score."""

    def __init__(self, cfg):
        """Init the parameters."""
        self.cfg = cfg
        self.cls_name = _get_class_names(cfg.PATH.CLASSES_PATH)
        self.cls_num = len(self.cfg.TRAIN.CLASSES)
        self.parsepredict = ParsePredict(self.cfg)
        print('use map')

    def init_parameters(self):
        self.gt_boxes = []
        self.gt_classes = []
        self.pred_boxes = []
        self.pred_classes = []
        self.pred_scores = []

    def cal_score(self, pre_labels, gt_labels=None, from_net=True):
        gt_labels = np.asarray(gt_labels.cpu())
        if from_net:
            pre_labels = self.parsepredict.parse_predict(pre_labels)
        for i, pre_label in enumerate(pre_labels):
            try:
                pre_label = np.asarray(pre_label)
                index = np.argsort(-pre_label[..., 0])
                pre_label = pre_label[index]
                pre_score = pre_label[..., 0]
                pre_cls = pre_label[..., 1]
                pre_box = pre_label[..., 2:]
            except:
                pre_score = []
                pre_cls = []
                pre_box = []

            self.pred_scores.append(pre_score)
            self.pred_classes.append(pre_cls)
            self.pred_boxes.append(pre_box)

            gt_label = gt_labels[gt_labels[..., 0] == i]
            gt_cls = gt_label[..., 1]
            gt_box = gt_label[..., 2:]

            self.gt_classes.append(gt_cls)
            self.gt_boxes.append(gt_box)

    def score_out(self):
        all_AP = self.eval_ap()
        mAP = 0.
        item_score = {}
        for key, value in all_AP.items():
            print('ap for {}: {}'.format(self.cfg.TRAIN.CLASSES[int(key)], value))
            if np.isnan(value):
                value = 0.
            mAP += float(value)
            item_score[self.cfg.TRAIN.CLASSES[int(key)]] = value
        mAP /= self.cls_num

        print("mAP=====>%.3f\n" % mAP)
        return mAP, item_score

    def eval_ap(self, iou_thread=0.5):
        """
        :param gt_boxes: list of 2d array,shape[(a,(x1,y1,x2,y2)),(b,(x1,y1,x2,y2))...]
        :param gt_labels: list of 1d array,shape[(a),(b)...],value is sparse label index
        :param pred_boxes: list of 2d array, shape[(m,(x1,y1,x2,y2)),(n,(x1,y1,x2,y2))...]
        :param pred_labels: list of 1d array,shape[(m),(n)...],value is sparse label index
        :param pred_scores: list of 1d array,shape[(m),(n)...]
        :param iou_thread: eg. 0.5
        :param num_cls: eg. 4, total number of class including background which is equal to 0
        :return: a dict containing average precision for each cls
        """
        all_ap = {}
        for label in range(self.cls_num):
            # get samples with specific label
            true_label_loc = [sample_labels == label for sample_labels in self.gt_classes]
            gt_single_cls = [sample_boxes[mask] for sample_boxes, mask in zip(self.gt_boxes, true_label_loc)]
            pred_label_loc = []
            for sample_labels in self.pred_classes:
                if sample_labels != []:
                    pred_label_loc.append(sample_labels == label)
                else:
                    pred_label_loc.append([])

            bbox_single_cls = []
            for sample_boxes, mask in zip(self.pred_boxes, pred_label_loc):
                if sample_boxes != []:
                    bbox_single_cls.append(sample_boxes[mask])
                else:
                    bbox_single_cls.append([])
            scores_single_cls = []
            for sample_scores, mask in zip(self.pred_scores, pred_label_loc):
                if sample_scores != []:
                    scores_single_cls.append(sample_scores[mask])
                else:
                    scores_single_cls.append([])

            fp = np.zeros((0,))
            tp = np.zeros((0,))
            scores = np.zeros((0,))
            total_gts = 0
            # loop for each sample
            for sample_gts, sample_pred_box, sample_scores in zip(gt_single_cls, bbox_single_cls, scores_single_cls):
                total_gts = total_gts + len(sample_gts)
                assigned_gt = []  # one gt can only be assigned to one predicted bbox
                # loop for each predicted bbox
                for index in range(len(sample_pred_box)):
                    scores = np.append(scores, sample_scores[index])
                    if len(sample_gts) == 0:  # if no gts found for the predicted bbox, assign the bbox to fp
                        fp = np.append(fp, 1)
                        tp = np.append(tp, 0)
                        continue
                    pred_box = np.expand_dims(sample_pred_box[index], axis=0)
                    iou = _iou_np(sample_gts, pred_box)
                    gt_for_box = np.argmax(iou, axis=0)
                    max_overlap = iou[gt_for_box, 0]
                    if max_overlap >= iou_thread and gt_for_box not in assigned_gt:
                        fp = np.append(fp, 0)
                        tp = np.append(tp, 1)
                        assigned_gt.append(gt_for_box)
                    else:
                        fp = np.append(fp, 1)
                        tp = np.append(tp, 0)
            # sort by score
            indices = np.argsort(-scores)
            fp = fp[indices]
            tp = tp[indices]
            # compute cumulative false positives and true positives
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            # compute recall and precision
            recall = tp / total_gts
            precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = self._compute_ap(recall, precision)
            all_ap[label] = ap
            # print(recall, precision)
        return all_ap

    def _compute_ap(self, recall, precision):
        """ Compute the average precision, given the recall and precision curves.
        Code originally from https://github.com/rbgirshick/py-faster-rcnn.
        # Arguments
            recall:    The recall curve (list).
            precision: The precision curve (list).
        # Returns
            The average precision as computed in py-faster-rcnn.
        """
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap
