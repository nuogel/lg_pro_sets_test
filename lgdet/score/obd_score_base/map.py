"""Calculate the F1 score."""
import numpy as np
from lgdet.util.util_get_cls_names import _get_class_names
from lgdet.util.util_iou import _iou_np, box_iou
from lgdet.registry import SCORES
from lgdet.postprocess.parse_factory import ParsePredict
import torch


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
        self.iouv = torch.linspace(0.5, 0.95, 10).to(self.cfg.TRAIN.DEVICE)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()

    def init_parameters(self):
        self.gt_boxes = []
        self.gt_classes = []
        self.pred_boxes = []
        self.pred_classes = []
        self.pred_scores = []
        self.stats = []

    def cal_score(self, pre_labels, test_data=None, from_net=True):
        gt_labels = test_data[1]
        # gt_labels = np.asarray(gt_labels.cpu())
        if from_net:
            pre_labels = self.parsepredict.parse_predict(pre_labels)
        for si, pre_label in enumerate(pre_labels):
            labels = gt_labels[gt_labels[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            if len(pre_label) == 0:
                if nl:
                    self.stats.append((torch.zeros(0, 10, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
            else:
                if nl:
                    correct = self._process_batch(pre_label, labels, self.iouv)
                else:
                    correct = torch.zeros(pre_label.shape[0], self.niou, dtype=torch.bool)

                self.stats.append((correct.cpu(), pre_label[:, 0].cpu(), pre_label[:, 1].cpu(), tcls))  # (correct, conf, pcls, tcls)

    def score_out(self):
        stats = [np.concatenate(x, 0) for x in zip(*self.stats)]  # to numpy
        if len(stats) and stats[0].any():
            p, r, ap, f1, ap_class = self._ap_per_class(*stats)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            # nt = np.bincount(stats[3].astype(np.int64), minlength=self.cls_num)  # number of targets per class
        else:
            mp, mr, map50, map = 0, 0, 0, 0
        print(('\n'+'%15s' * 4) % ('meanP', 'meanR', 'mAP@.5', 'mAP@.5:.95'))
        print(('%15f' * 4)% (mp, mr, map50, map))
        return map50, {'map50:95': map, 'mean_precision': mp, 'mean_recall': mr}

    def _ap_per_class(self, tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=()):
        """ Compute the average precision, given the recall and precision curves.
        Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
        # Arguments
            tp:  True positives (nparray, nx1 or nx10).
            conf:  Objectness value from 0-1 (nparray).
            pred_cls:  Predicted object classes (nparray).
            target_cls:  True object classes (nparray).
            plot:  Plot precision-recall curve at mAP@0.5
            save_dir:  Plot save directory
        # Returns
            The average precision as computed in py-faster-rcnn.
        """

        # Sort by objectness
        i = np.argsort(-conf)
        tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

        # Find unique classes
        unique_classes = np.unique(target_cls)
        nc = unique_classes.shape[0]  # number of classes, number of detections

        # Create Precision-Recall curve and compute AP for each class
        px, py = np.linspace(0, 1, 1000), []  # for plotting
        ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
        for ci, c in enumerate(unique_classes):
            i = pred_cls == c
            n_l = (target_cls == c).sum()  # number of labels
            n_p = i.sum()  # number of predictions

            if n_p == 0 or n_l == 0:
                continue
            else:
                # Accumulate FPs and TPs
                fpc = (1 - tp[i]).cumsum(0)
                tpc = tp[i].cumsum(0)

                # Recall
                recall = tpc / (n_l + 1e-16)  # recall curve
                r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

                # Precision
                precision = tpc / (tpc + fpc)  # precision curve
                p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

                # AP from recall-precision curve
                for j in range(tp.shape[1]):
                    ap[ci, j], mpre, mrec = self._compute_ap(recall[:, j], precision[:, j])
                    if plot and j == 0:
                        py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

        # Compute F1 (harmonic mean of precision and recall)
        f1 = 2 * p * r / (p + r + 1e-16)

        i = f1.mean(0).argmax()  # max F1 index
        return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')

    def _process_batch(self, detections, labels, iouv):
        """
        Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6])  conf, class, x1, y1, x2, y2
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            correct (Array[N, 10]), for 10 IoU levels
        """
        correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
        iou = box_iou(labels[:, 1:], detections[:, 2:])
        x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 1]))  # IoU above threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            matches = torch.Tensor(matches).to(iouv.device)
            correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
        return correct

    def _compute_ap(self, recall, precision):
        """ Compute the average precision, given the recall and precision curves
        # Arguments
            recall:    The recall curve (list)
            precision: The precision curve (list)
        # Returns
            Average precision, precision curve, recall curve
        """

        # Append sentinel values to beginning and end
        mrec = np.concatenate(([0.], recall, [recall[-1] + 0.01]))
        mpre = np.concatenate(([1.], precision, [0.]))

        # Compute the precision envelope
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

        # Integrate area under curve
        method = 'interp'  # methods: 'continuous', 'interp'
        if method == 'interp':
            x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
            ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
        else:  # 'continuous'
            i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

        return ap, mpre, mrec
