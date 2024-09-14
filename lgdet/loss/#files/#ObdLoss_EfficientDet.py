import numpy as np
import torch
import torch.nn as nn
from lgdet.util.util_iou import iou_xyxy, xywh2xyxy


class EfficientDetLoss():
    def __init__(self, cfg):
        self.bceloss = torch.nn.BCELoss(reduction='none')

    def Loss_Call(self, predicted, dataset, losstype=None):

        classifications, regressions, anchors_xywh = predicted
        annotations = dataset[1]
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = 0.
        regression_losses = 0.

        anchor_xyxy = xywh2xyxy(anchors_xywh[0, :, :])

        anchor_ctr_x = anchors_xywh[0, :, 0]
        anchor_ctr_y = anchors_xywh[0, :, 1]
        anchor_widths = anchors_xywh[0, :, 2]
        anchor_heights = anchors_xywh[0, :, 3]

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = torch.Tensor(annotations[j]).to(anchors_xywh.device)
            bbox_annotation = bbox_annotation[bbox_annotation[:, 0] != -1]

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)
            IoU = iou_xyxy(anchor_xyxy, bbox_annotation[:, 1:])  # num_anchors x num_annotations

            IoU_max, IoU_argmax = torch.max(IoU, dim=1)  # num_anchors x 1

            # import pdb
            # pdb.set_trace()

            # compute the loss for classification
            class_targets = torch.ones(classification.shape) * -1
            class_targets = class_targets.to(anchor_xyxy.device)

            class_targets[torch.lt(IoU_max, 0.4), :] = 0

            positive_indices = torch.ge(IoU_max, 0.5)

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            class_targets[positive_indices, :] = 0
            class_targets[positive_indices, assigned_annotations[positive_indices, 0].long()] = 1

            alpha_factor = torch.ones(class_targets.shape) * alpha
            alpha_factor = alpha_factor.to(anchor_xyxy.device)
            alpha_factor = torch.where(torch.eq(class_targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(class_targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
            a, A, b = class_targets.min(), class_targets.max(), classification.max()
            # bce = -(class_targets * torch.log(classification) + (1.0 - class_targets) * torch.log(1.0 - classification))
            bce = self.bceloss(classification, class_targets)
            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * bce

            cls_loss = torch.where(torch.ne(class_targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).to(anchor_xyxy.device))
            classification_losses += (cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0))

            # classification_losses.append(cls_loss.sum())

            # compute the loss for regression

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_heights = assigned_annotations[:, 4] - assigned_annotations[:, 2]
                gt_ctr_x = assigned_annotations[:, 1] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 2] + 0.5 * gt_heights

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                targets = targets / torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).to(anchors_xywh.device)

                negative_indices = ~positive_indices

                regression_diff = torch.abs(targets - regression[positive_indices, :])
                # regression_loss = torch.pow(regression_diff, 2)
                regression_loss = torch.where(torch.le(regression_diff, 1.0 / 9.0), 0.5 * 9.0 * torch.pow(regression_diff, 2), regression_diff - 0.5 / 9.0)
                regression_losses += regression_loss.mean()

        class_loss = classification_losses / batch_size
        loc_loss = regression_losses / batch_size
        return class_loss, loc_loss
