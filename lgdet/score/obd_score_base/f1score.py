"""Calculate the F1 score."""
import numpy as np
import torch
from lgdet.util.util_get_cls_names import _get_class_names
from lgdet.util.util_iou import iou_xyxy
from lgdet.registry import SCORES
from lgdet.postprocess.parse_factory import ParsePredict


@SCORES.registry()
class FnScore:
    """Calculate the F1 score."""

    def __init__(self, cfg):
        """Init the parameters."""
        self.cfg = cfg
        self.cls_name = _get_class_names(cfg.PATH.CLASSES_PATH)
        self.cls_num = len(self.cfg.TRAIN.CLASSES)
        self.parsepredict = ParsePredict(self.cfg)
        print('use f-score')

    def init_parameters(self):
        self.true_positive = {}
        self.false_positive = {}
        self.pre_scores = {}
        for i in range(self.cls_num):
            self.true_positive[i] = []
            self.false_positive[i] = []
            self.pre_scores[i] = []
        self.gt_obj_num = np.zeros(self.cls_num)

    def cal_score(self, pre_labels, gt_labels,
                  from_net=True):  # did not merge the self.get_labels_txt, because from net,don't need that function.
        """
        Calculate the TP & FP.

        :param pre_labels: prediction labels
        :param gt_labels: ground truth labels
        :param from_net : pre_labels from net work
        :return: TP FP object numbers
        """
        if from_net:
            pre_labels = self.parsepredict.parse_predict(pre_labels)
        if pre_labels != [[]] and pre_labels != []:
            for i, pre_label in enumerate(pre_labels):  # calculate every image
                self._cal_per_image(pre_label, gt_labels[gt_labels[..., 0] == i])
        # else:
        #     print('predicted labels is None.')

    def _cal_per_image(self, pre_label, gt_label, iou_thresh=0.5):
        """
        Calculate TP FP for one image.

        :param pre_label: prediction labels of one image
        :param gt_label: ground truth labels of one image
        :param iou_thresh: the threshold of iou
        :return: P FP object numbers of one image
        """
        label_used = np.zeros(len(gt_label))

        for cls in range(self.cls_num):
            for one_pre_box in pre_label:
                pre_cls = one_pre_box[1]
                if pre_cls != cls:
                    continue
                max_iou = -1
                _id = None
                for i, lab in enumerate(gt_label):
                    lab = lab[1:]
                    cls_gt = int(lab[0].cpu())
                    if cls_gt != cls:
                        continue
                    pre_box = torch.Tensor(one_pre_box[2:]).unsqueeze(0).to(self.cfg.TRAIN.DEVICE)
                    gt_box = lab[1:5].unsqueeze(0)
                    iou = iou_xyxy(pre_box, gt_box)[0][0]  # to(anchors.device))
                    if iou > max_iou:
                        max_iou = iou
                        _id = i
                if max_iou > iou_thresh and label_used[_id] == 0:
                    self.true_positive[cls].append(1)
                    self.false_positive[cls].append(0)
                    label_used[_id] = 1
                else:
                    self.true_positive[cls].append(0)
                    self.false_positive[cls].append(1)
                self.pre_scores[cls].append(one_pre_box[0])
        for lab in gt_label:
            self.gt_obj_num[int(lab[1])] += 1

    def score_out(self):
        f1score, prec, rec = self.F1_SCORE()

        f1score[f1score == -1] = 0
        F1Score = f1score.mean()

        f1score_dict = dict(zip(self.cfg.TRAIN.CLASSES, f1score))
        prec_dict = dict(zip(self.cfg.TRAIN.CLASSES, prec))
        rec_dict = dict(zip(self.cfg.TRAIN.CLASSES, rec))

        f1score_txt = ['%0.2f' % score for score in f1score]
        prec_txt = ['%0.2f' % score for score in prec]
        rec_txt = ['%0.2f' % score for score in rec]

        print('%30s' * 4 % ('classes', 'f1-score', 'precision', 'recall')+'\n')
        for cls, f, p, r in zip(self.cfg.TRAIN.CLASSES, f1score_txt, prec_txt, rec_txt):
            print('%30s'*4 % (cls,f , p, r))
        f1infos = 'f1score:%0.4f\nf1sc: %s\nprec: %s\nreca: %s' % (F1Score, f1score_txt, prec_txt, rec_txt)
        print(f1infos)
        self.cfg.logger.info(f1infos)
        return F1Score, {'f1score': f1score_dict, 'prec_dict': prec_dict, 'rec_dict': rec_dict}

    def F1_SCORE(self, beta=1):
        """
        Calculate the F1 score.

        :param beta: beta
        :return: F1 score
        """
        f1_sore = -1 * np.ones(self.cls_num, dtype=np.float32)
        prec = -1 * np.ones(self.cls_num, dtype=np.float32)
        rec = -1 * np.ones(self.cls_num, dtype=np.float32)
        for i in range(self.cls_num):
            if self.gt_obj_num[i] == 0:
                # TODO: BUG, when there is a FP about this class. how ?, But ,gt_obj_num is hardly tobe Zero.
                continue
            tp = np.sum(np.asarray(self.true_positive[i], np.float32))
            fp = np.sum(np.asarray(self.false_positive[i], np.float32))

            if tp == 0 and fp == 0:
                prec[i] = 0
                rec[i] = 0
                f1_sore[i] = 0
                continue
            prec[i] = tp / (tp + fp + 1e-16)
            rec[i] = tp / (self.gt_obj_num[i] + 1e-16)
            f1_sore[i] = (1 + beta ** 2) * (prec[i] * rec[i]) / (beta ** 2 * prec[i] + rec[i] + 1e-16)
        return f1_sore, prec, rec
