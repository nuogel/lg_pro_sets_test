import torch.nn.functional as F
import torch
from lgdet.util.util_iou import xyxy2xywh, iou_xywh
from lgdet.loss.loss_base.focal_loss import FocalLoss_lg, FocalLoss


class RETINANETLOSS():
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.TRAIN.DEVICE
        self.num_cls = self.cfg.TRAIN.CLASSES_NUM
        # self.focalloss = FocalLoss_lg()
        self.focalloss = FocalLoss(bceonly=False, add_logist=True, reduction='sum')

    def Loss_Call(self, predictions, targets, kwargs):  # predictions with sigmoid()
        pre_cls, pre_loc, anc_xywh = predictions
        # print(pre_cls.shape, pre_loc.shape, anc_xywh.shape)
        # pre_cls = torch.clamp(pre_cls, 1e-5, 1.0 - 1e-4)

        with torch.no_grad():
            batchsize = pre_loc.shape[0]
            _, gt_labels, infos = targets
            gt_loc, gt_cls, pos_mask, neg_mask = [], [], [], []
            for i in range(batchsize):
                gt_i = gt_labels[gt_labels[..., 0] == i]
                lab = gt_i[..., 1].long()
                box = gt_i[..., 2:]
                box_xywh = xyxy2xywh(box)
                _gt_cls, _gt_loc_xywh, _pos_mask, _neg_mask = self._assign_priors(pre_cls[i], box_xywh, lab, anc_xywh[i])
                _gt_loc_xywh = self._encode_bbox(_gt_loc_xywh, anc_xywh[i])
                gt_loc.append(_gt_loc_xywh)
                gt_cls.append(_gt_cls)
                pos_mask.append(_pos_mask)
                neg_mask.append(_neg_mask)
            [gt_loc, gt_cls, pos_mask, neg_mask] = [torch.stack(xx, dim=0) for xx in [gt_loc, gt_cls, pos_mask, neg_mask]]
            pos_neg_mask = pos_mask | neg_mask
            gt_loc = gt_loc[pos_mask, :].reshape(-1, 4)
            pos_num = pos_mask.sum()

        # cls_loss, pos_loss, neg_loss = self.focalloss(pre_cls, gt_cls, pos_mask, neg_mask, reduction='sum')  # FOCALLOSS_LG
        cls_loss, pos_loss, neg_loss = self.focalloss(pre_cls[pos_neg_mask], gt_cls[pos_neg_mask])
        pre_loc = pre_loc[pos_mask, :].reshape(-1, 4)
        loc_loss = F.smooth_l1_loss(pre_loc, gt_loc)*2  #
        # loc_loss = loc_loss.mean(-1).sum()

        # pos_loss /= pos_num/self.num_cls
        # neg_loss /= (pos_num)
        cls_loss = cls_loss/pos_num
        total_loss = cls_loss + loc_loss
        # total_loss = pos_loss + neg_loss + loc_loss
        # total_loss*=5
        # metrics:

        with torch.no_grad():
            pre_cls=pre_cls.sigmoid()
            cls_p = (pre_cls[pos_mask].argmax(-1) == gt_cls[pos_mask].argmax(-1)).float().mean().item()
            pos_sc = pre_cls[pos_mask].max(-1)[0].mean().item()
            pos_t = (pre_cls[pos_mask].max(-1)[0] > self.cfg.TEST.SCORE_THRESH).float().mean().item()
            neg_sc = pre_cls[neg_mask].max(-1)[0].mean().item()
            neg_t = (pre_cls[neg_mask].max(-1)[0] > self.cfg.TEST.SCORE_THRESH).float().sum().item()

            metrics = {
                'pos_num': pos_num.item(),
                'pos_sc': pos_sc,
                'neg_sc': neg_sc,
                'obj>t': pos_t,
                'nob_t': neg_t,
                'cls_p': cls_p,
                'cls_loss': cls_loss.item(),
                # 'pos_loss': pos_loss.item(),
                # 'neg_loss': neg_loss.item(),
                'loc_loss': loc_loss.item(),
            }
        return {'total_loss': total_loss, 'metrics': metrics}

    def _assign_priors(self, pre_cls, gt_boxes, gt_labels, corner_form_priors):
        ious = iou_xywh(corner_form_priors, gt_boxes, type='N2N')
        best_target_per_prior, best_target_per_prior_index = ious.max(1)

        # #Faster rcnn中的第一条Anchor分配规则是如果最大IoU也没有大于0.5，
        # 则这个最大IoU的Anchor也设为正样本。但是在遍历COCO数据集后发现，这种情况非常少见，
        # 因此我们不使用第一条Anchor分配规则。这样相当于这部分object没有用于训练，
        # 但由于数量很少，对模型的性能表现不会产生影响。
        best_prior_per_target, best_prior_per_target_index = ious.max(0)
        for target_index, prior_index in enumerate(best_prior_per_target_index):
            best_target_per_prior_index[prior_index] = target_index
        # 2.0 is used to make sure every target has a prior assigned
        best_target_per_prior.index_fill_(0, best_prior_per_target_index, 1.)  # fill 2>1.
        # size: num_priors

        gt_cls = torch.zeros_like(pre_cls).to(self.device)
        pos_mask = best_target_per_prior >= 0.5 # official is 0.5 ,but the rest will not that good for nms.
        neg_mask = best_target_per_prior < 0.5 # official is 0.4 ,but the rest will not that good for nms.

        best_target_per_prior_index_pose = best_target_per_prior_index[pos_mask]
        # gt_cls[neg_mask, -1] = 1  # background set to one hot [0, ..., 1]
        assigned_annotations = gt_labels[best_target_per_prior_index_pose]

        # gt_cls[pos_mask, :] = 0
        # print(gt_cls.shape, gt_cls.device, pos_mask.shape,  pos_mask.device)
        gt_cls[pos_mask, assigned_annotations.long()] = 1
        gt_loc = gt_boxes[best_target_per_prior_index]
        return gt_cls, gt_loc, pos_mask, neg_mask

    def _encode_bbox(self, xywh_boxes, xywh_priors):
        encode_target = torch.cat([(xywh_boxes[..., :2] - xywh_priors[..., :2]) / xywh_priors[..., 2:] / 0.1,
                                   torch.log(xywh_boxes[..., 2:] / xywh_priors[..., 2:]) / 0.2],
                                  dim=- 1)
        return encode_target
