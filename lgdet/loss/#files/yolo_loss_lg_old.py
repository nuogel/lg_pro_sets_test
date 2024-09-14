"""Loss calculation based on yolo."""
import torch
import numpy as np
from lgdet.util.util_iou import iou_xywh
from lgdet.util.util_get_cls_names import _get_class_names
from lgdet.postprocess.parse_factory import ParsePredict


class YoloLoss:
    # pylint: disable=too-few-public-methods
    """Calculate loss."""

    def __init__(self, cfg):
        """Init."""
        #
        self.cfg = cfg
        self.device = self.cfg.TRAIN.DEVICE

        self.anchors = torch.Tensor(cfg.TRAIN.ANCHORS)
        self.anc_num = cfg.TRAIN.FMAP_ANCHOR_NUM
        self.cls_num = cfg.TRAIN.CLASSES_NUM
        self.one_test = cfg.TEST.ONE_TEST

        self.batch_size = cfg.TRAIN.BATCH_SIZE

        self.mseloss = torch.nn.MSELoss(reduction='sum')
        self.bceloss = torch.nn.BCELoss(reduction='none')
        self.class_name = _get_class_names(cfg.PATH.CLASSES_PATH)

        self.parsepredict = ParsePredict(cfg)
        self.multiply_area_scale = False  # whether multiply loss to area_scale.

    def _reshape_labels(self, labels, grid_xy, shape, f_id):
        """
        Reshape the labels.

        :param labels: labels from training data
        :param grid_xy: the matrix of the grid numbers
        :return: labels_obj, labels_cls, lab_loc_xy, lab_loc_wh, labels_boxes, area_scal
        """
        mask = np.arange(self.anc_num) + self.anc_num * f_id
        anchors = self.anchors[mask]
        B = grid_xy.shape[0]
        labels_obj = torch.zeros([B, shape[0], shape[1],
                                  self.anc_num, 1]).to(self.cfg.TRAIN.DEVICE)
        labels_loc = torch.zeros([B, shape[0], shape[1],
                                  self.anc_num, 4]).to(self.cfg.TRAIN.DEVICE)
        labels_cls = torch.zeros([B, shape[0], shape[1],
                                  self.anc_num, self.cls_num]).to(self.cfg.TRAIN.DEVICE)
        labels_boxes = []
        for batch_idx in range(B):
            labs = labels[labels[..., 0] == batch_idx][..., 1:]
            lab_boxes = []
            for lab in labs:
                box_xy = (lab[1:3] + lab[3:5]) / 2
                box_wh = (lab[3:5] - lab[1:3])
                boxes = torch.cat([box_xy, box_wh])
                box_center = (box_xy * torch.Tensor([shape[1], shape[0]]).to(self.cfg.TRAIN.DEVICE)).long()
                anc = torch.cat([torch.zeros_like(anchors), anchors], 1)
                box_iou = torch.cat([torch.Tensor([0, 0]).to(self.cfg.TRAIN.DEVICE), box_wh])
                iou = iou_xywh(anc.to(self.cfg.TRAIN.DEVICE), box_iou.to(self.cfg.TRAIN.DEVICE), type='N21')
                iou_max = torch.max(iou, 0)
                anc_idx = iou_max[1].item()
                lab_boxes.append(boxes)
                # print(box_center, boxes)

                labels_obj[batch_idx, box_center[1], box_center[0], anc_idx, 0] = 1
                labels_loc[batch_idx, box_center[1], box_center[0], anc_idx] = boxes
                labels_cls[batch_idx, box_center[1], box_center[0], anc_idx, lab[0].long()] = 1
            labels_boxes.append(lab_boxes)

        lab_loc_xy = labels_loc[..., 0:2]
        lab_loc_wh = labels_loc[..., 2:4]
        # count the scale of  w*h, in order to count area_scal*wh
        if self.multiply_area_scale:
            area_scale = (1 - torch.sqrt(lab_loc_wh[..., 0] * lab_loc_wh[..., 1])).unsqueeze(-1).expand_as(lab_loc_wh)
        else:
            area_scale = 1.0
        # print(area_scal.shape, area_scal[0, 14, 39, 12:])
        lab_loc_xy = lab_loc_xy * torch.Tensor([shape[1], shape[0]]).to(self.cfg.TRAIN.DEVICE) - grid_xy
        anchor_ch = anchors.view(1, 1, 1, self.anc_num, 2).expand(1, shape[0], shape[1], self.anc_num, 2).to(self.cfg.TRAIN.DEVICE)
        lab_loc_wh = lab_loc_wh / anchor_ch
        lab_loc_wh = torch.log(torch.clamp(lab_loc_wh, 1e-9, 1e9))

        return labels_obj, labels_cls, lab_loc_xy, lab_loc_wh, labels_boxes, area_scale

        # def _reshape_predict(self, f_map, tolabel=False):
        #     # pylint: disable=no-self-use
        #     """
        #     Reshape the predict, or reshape it to label shape.
        #
        #     :param predict: out of net
        #     :param tolabel: True or False to label shape
        #     :return:
        #     """
        #     ###1: pre deal feature mps
        #     obj_pred, cls_perd, loc_pred = torch.split(f_map,
        #                                                [self.anc_num, self.anc_num * self.cls_num, self.anc_num * 4], 3)
        #     pre_obj = obj_pred.sigmoid()
        #     cls_reshape = torch.reshape(cls_perd, (-1, 4))
        #     cls_pred_prob = torch.softmax(cls_reshape, -1)
        #     pre_cls = torch.reshape(cls_pred_prob, cls_perd.shape)
        #     pre_loc = loc_pred
        #     # pre_obj:[N,self.cfg.BOX_HEIGHT,self.cfg.BOX_WIDTH,self.anc_num]
        #     # pre_cls:[N,self.cfg.BOX_HEIGHT,self.cfg.BOX_WIDTH,64]
        #     # pre_loc:[N,self.cfg.BOX_HEIGHT,self.cfg.BOX_WIDTH,64]
        #     batch_size = pre_obj.shape[0]
        #     shape = pre_obj.shape[1:3]
        #
        #     pre_obj = pre_obj.unsqueeze(-1)
        #     pre_cls = pre_cls.reshape([batch_size, shape[0], shape[1], self.anc_num, self.cls_num])
        #
        #     # reshape the pre_loc
        #     pre_loc = pre_loc.reshape([batch_size, shape[0], shape[1], self.anc_num, 4])
        #     pre_loc_xy = pre_loc[..., 0:2].sigmoid()
        #     pre_loc_wh = pre_loc[..., 2:4]
        #
        #     grid_x = torch.arange(0, shape[1]).view(-1, 1).repeat(1, shape[0]).unsqueeze(2).permute(1, 0, 2)
        #     grid_y = torch.arange(0, shape[0]).view(-1, 1).repeat(1, shape[1]).unsqueeze(2)
        #     grid_xy = torch.cat([grid_x, grid_y], 2).unsqueeze(2).unsqueeze(0). \
        #         expand(1, shape[0], shape[1], self.anc_num, 2).expand_as(pre_loc_xy).type(torch.cuda.FloatTensor)
        #
        #     # prepare gird xy
        #     box_ch = torch.Tensor([shape[1], shape[0]]).to(self.cfg.TRAIN.DEVICE)
        #     pre_realtive_xy = (pre_loc_xy + grid_xy) / box_ch
        #     anchor_ch = self.anchors.view(1, 1, 1, self.anc_num, 2).expand(1, shape[0], shape[1], self.anc_num, 2).to(self.cfg.TRAIN.DEVICE)
        #     pre_realtive_wh = pre_loc_wh.exp() * anchor_ch
        #
        #     pre_relative_box = torch.cat([pre_realtive_xy, pre_realtive_wh], -1)
        #
        #     if tolabel:
        #         return pre_obj, pre_cls, pre_relative_box
        #     return pre_obj, pre_cls, pre_relative_box, pre_loc_xy, pre_loc_wh, grid_xy, shape

    def _ignore(self, pre_loc, lab_boxes):
        # pylint: disable=no-self-use
        """
        Count the ignore mask.

        :param pre_loc:
        :param lab_boxes:
        :return:
        """
        batch = pre_loc.shape[0]
        _ignore_mask = []
        for i in range(batch):
            b_box = lab_boxes[i]
            b_box = torch.stack(b_box, 0)
            iou = iou_xywh(pre_loc[i], b_box, type='N2N_yolo')
            iou_max = torch.max(iou, 0)
            # print(iou_max[0][15, 31:34])
            ignore_iou = torch.lt(iou_max[0], 0.6)
            # print(ignore_iou[15, 31:34])
            _ignore_mask.append(ignore_iou)

        ignore_mask = torch.stack(_ignore_mask, 0)
        # print(ignore_mask[0, 15, 31:34])

        return ignore_mask

    # def raw_loss_cal(self, predict, labels, losstype=None):
    #     """Calculate the loss."""
    #     # pre_obj:[N,self.cfg.BOX_HEIGHT,self.cfg.BOX_WIDTH,self.anc_num]
    #     # pre_cls:[N,self.cfg.BOX_HEIGHT,self.cfg.BOX_WIDTH,self.anc_num,4]
    #     # pre_loc:[N,self.cfg.BOX_HEIGHT,self.cfg.BOX_WIDTH,self.anc_num,4]
    #     pre_obj, pre_cls, pre_relative_box, pre_loc_xy, pre_loc_wh, grid_xy, shape = \
    #         self._reshape_predict(predict)
    #     # lab_obj:[N,self.cfg.BOX_HEIGHT,self.cfg.BOX_WIDTH,self.anc_num]
    #     # lab_cls:[N,self.cfg.BOX_HEIGHT,self.cfg.BOX_WIDTH,self.anc_num,4]
    #     # lab_loc:[N,self.cfg.BOX_HEIGHT,self.cfg.BOX_WIDTH,self.anc_num,4]
    #     lab_obj, lab_cls, lab_loc_xy, lab_loc_wh, lab_boxes, area_scal = \
    #         self._reshape_labels(labels, grid_xy, shape)
    #
    #     # '''to be a mask:'''
    #     # '''
    #     # # obj_mask:[N,self.cfg.BOX_HEIGHT,self.cfg.BOX_WIDTH,self.anc_num,1]
    #     # obj_mask = torch.eq(lab_obj, 5)
    #     # # ignore_mask:[N,self.cfg.BOX_HEIGHT,self.cfg.BOX_WIDTH,self.anc_num,1]
    #     # ignore_mask = self._ignore(pre_relative_box, lab_boxes)
    #     # ignore_mask = ignore_mask.unsqueeze(-1).type(torch.cuda.FloatTensor)
    #     # ignore_mask = torch.eq(ignore_mask, 1)
    #     # ignore_mask = (1 - obj_mask) * ignore_mask
    #     #
    #     # obj_mask_2 = obj_mask.expand_as(pre_loc_xy)
    #     # obj_mask.expand_as(pre_cls) = obj_mask.expand_as(pre_cls)
    #     #
    #     # obj_loss = self.mseloss(pre_obj[obj_mask], lab_obj[obj_mask]) * 25
    #     # noobj_loss = self.mseloss(pre_obj[ignore_mask], lab_obj[ignore_mask])
    #     # # noobj_loss = obj_loss*0
    #     # lxy_loss = self.mseloss(pre_loc_xy[obj_mask_2], lab_loc_xy[obj_mask_2])
    #     # lwh_loss = self.mseloss(pre_loc_wh[obj_mask_2], lab_loc_wh[obj_mask_2])
    #     # cls_loss = self.mseloss(pre_cls[obj_mask.expand_as(pre_cls)],
    #     # lab_cls[obj_mask.expand_as(pre_cls)])
    #     # loc_loss = lxy_loss + lwh_loss
    #     # '''
    #     #
    #     # '''to be a matrx:'''
    #     # '''
    #     # obj_mask = torch.eq(lab_obj, 1).type(torch.cuda.FloatTensor)
    #     obj_mask = lab_obj
    #     # ignore_mask:[N,self.cfg.BOX_HEIGHT,self.cfg.BOX_WIDTH,self.anc_num,1]
    #     ignore_mask = self._ignore(pre_relative_box, lab_boxes)
    #     ignore_mask = ignore_mask.unsqueeze(-1).type(torch.cuda.FloatTensor)
    #     ignore_mask = (1 - obj_mask) * ignore_mask
    #     # print(ignore_mask.sum())
    #
    #     obj_mask_2 = obj_mask.expand_as(pre_loc_xy)
    #
    #     if losstype == 'focalloss':
    #         # FOCAL loss
    #         alpha = 0.25
    #         obj_loss = alpha * pow((torch.ones_like(pre_obj) - pre_obj), 2) * self.bceloss(pre_obj * obj_mask,
    #                                                                                        lab_obj * obj_mask)
    #         noobj_loss = (1 - alpha) * pow(pre_obj, 2) * self.bceloss(pre_obj * ignore_mask, lab_obj * ignore_mask)
    #         obj_loss = torch.sum(obj_loss) / self.batch_size
    #         noobj_loss = torch.sum(noobj_loss) / self.batch_size
    #     elif losstype == 'mse' or losstype is None:
    #         # nomal loss
    #         obj_loss = self.mseloss(pre_obj * obj_mask, lab_obj * obj_mask) * 25 / self.batch_size
    #         noobj_loss = self.mseloss(pre_obj * ignore_mask, lab_obj * ignore_mask) * 1 / self.batch_size
    #
    #     lxy_loss = self.mseloss(pre_loc_xy * obj_mask_2, lab_loc_xy * obj_mask_2) / self.batch_size
    #     lwh_loss = self.mseloss(pre_loc_wh * area_scal * obj_mask_2,
    #                             lab_loc_wh * area_scal * obj_mask_2) / self.batch_size
    #     cls_loss = self.mseloss(pre_cls * obj_mask.expand_as(pre_cls),
    #                             lab_cls * obj_mask.expand_as(pre_cls)) / self.batch_size
    #
    #     return noobj_loss, obj_loss, cls_loss, lxy_loss + lwh_loss

    def _loss_cal_one_Fmap(self, f_map, f_id, labels, kwargs):
        """Calculate the loss."""
        loc_losstype, obj_losstype = kwargs['losstype']
        metrics = {}
        pre_obj, pre_cls, pre_relative_box, pre_loc_xy, pre_loc_wh, grid_xy, shape = \
            self.parsepredict._parse_yolo_predict_fmap_old(f_map, f_id)

        lab_obj, lab_cls, lab_loc_xy, lab_loc_wh, lab_boxes, area_scal = \
            self._reshape_labels(labels, grid_xy, shape, f_id)

        # '''to be a mask:'''
        # '''
        # # obj_mask:[N,self.cfg.BOX_HEIGHT,self.cfg.BOX_WIDTH,self.anc_num,1]
        # obj_mask = torch.eq(lab_obj, 5)
        # # ignore_mask:[N,self.cfg.BOX_HEIGHT,self.cfg.BOX_WIDTH,self.anc_num,1]
        # ignore_mask = self._ignore(pre_relative_box, lab_boxes)
        # ignore_mask = ignore_mask.unsqueeze(-1).type(torch.cuda.FloatTensor)
        # ignore_mask = torch.eq(ignore_mask, 1)
        # ignore_mask = (1 - obj_mask) * ignore_mask
        #
        # obj_mask_2 = obj_mask.expand_as(pre_loc_xy)
        # obj_mask.expand_as(pre_cls) = obj_mask.expand_as(pre_cls)
        #
        # obj_loss = self.mseloss(pre_obj[obj_mask], lab_obj[obj_mask]) * 25
        # noobj_loss = self.mseloss(pre_obj[ignore_mask], lab_obj[ignore_mask])
        # # noobj_loss = obj_loss*0
        # lxy_loss = self.mseloss(pre_loc_xy[obj_mask_2], lab_loc_xy[obj_mask_2])
        # lwh_loss = self.mseloss(pre_loc_wh[obj_mask_2], lab_loc_wh[obj_mask_2])
        # cls_loss = self.mseloss(pre_cls[obj_mask.expand_as(pre_cls)],
        # lab_cls[obj_mask.expand_as(pre_cls)])
        # loc_loss = lxy_loss + lwh_loss
        # '''
        #
        # '''to be a matrx:'''
        # '''
        # obj_mask = torch.eq(lab_obj, 1).type(torch.cuda.FloatTensor)
        obj_mask = lab_obj
        # ignore_mask:[N,self.cfg.BOX_HEIGHT,self.cfg.BOX_WIDTH,self.anc_num,1]
        ignore_mask = self._ignore(pre_relative_box, lab_boxes)
        ignore_mask = ignore_mask.unsqueeze(-1).type(torch.cuda.FloatTensor)
        # ignore_mask = 1.0
        ignore_mask = (1 - obj_mask) * ignore_mask
        # print(ignore_mask.sum())

        obj_mask_2 = obj_mask.expand_as(pre_loc_xy)

        '''
        Debug code.
        if pre_obj.shape[1] > 15:
            i_0, i_1, i_2 = 0, 14, 37  # 0, 7, 18
            print('pre_obj[i_0, i_1, i_2]:', pre_obj[i_0, i_1, i_2])
            print('lab_obj[i_0, i_1, i_2]:', lab_obj[i_0, i_1, i_2])
            # print('pre_cls[i_0, i_1, i_2]', pre_cls[i_0, i_1, i_2])
            # print('GT_cls[i_0, i_1, i_2]', lab_cls[i_0, i_1, i_2])
            # print('pre_loc_xy', pre_loc_xy[i_0, i_1, i_2])
            # print('lab_loc_xy', lab_loc_xy[i_0, i_1, i_2])
            print('obj_mask', obj_mask[i_0, i_1, i_2])
            print('NOobj_mask', ignore_mask[i_0, i_1, i_2])
        '''

        if obj_losstype == 'focalloss':
            # FOCAL loss
            alpha = 0.25
            obj_loss = alpha * pow((torch.ones_like(pre_obj) - pre_obj), 2) * self.bceloss(pre_obj * obj_mask,
                                                                                           lab_obj * obj_mask)
            noobj_loss = (1 - alpha) * pow(pre_obj, 2) * self.bceloss(pre_obj * ignore_mask, lab_obj * ignore_mask)
            obj_loss = torch.sum(obj_loss) / self.batch_size
            noobj_loss = torch.sum(noobj_loss) / self.batch_size

        lxy_loss = self.mseloss(pre_loc_xy * obj_mask_2, lab_loc_xy * obj_mask_2) / self.batch_size
        lwh_loss = self.mseloss(pre_loc_wh * area_scal * obj_mask_2,
                                lab_loc_wh * area_scal * obj_mask_2) / self.batch_size
        cls_loss = self.mseloss(pre_cls * obj_mask.expand_as(pre_cls),
                                lab_cls * obj_mask.expand_as(pre_cls)) / self.batch_size
        loc_loss = lxy_loss + lwh_loss

        obj_mask = obj_mask.squeeze() > 0.
        noobj_mask = ignore_mask.squeeze() > 0.

        # if obj_losstype == 'focalloss':
        #     # FOCAL loss
        #     alpha = 0.25
        #     obj_loss = alpha * pow((1. - pre_obj[obj_mask]), 2) * self.bceloss(pre_obj[obj_mask], lab_obj[obj_mask])
        #     noobj_loss = (1 - alpha) * pow(pre_obj[noobj_mask], 2) * self.bceloss(pre_obj[noobj_mask], lab_obj[noobj_mask])
        #     obj_loss = torch.sum(obj_loss) / self.batch_size
        #     noobj_loss = torch.sum(noobj_loss) / self.batch_size
        #
        # lxy_loss = self.mseloss(pre_loc_xy[obj_mask], lab_loc_xy[obj_mask]) / self.batch_size
        # lwh_loss = self.mseloss(pre_loc_wh[obj_mask], lab_loc_wh[obj_mask]) / self.batch_size
        # cls_loss = self.mseloss(pre_cls[obj_mask], lab_cls[obj_mask]) / self.batch_size
        # loc_loss = lxy_loss + lwh_loss

        obj_num = (obj_mask > 0).sum()
        metrics = {}
        obj_sc = pre_obj[obj_mask].mean().item() if obj_num > 0. else 0.
        noobj_sc = pre_obj[noobj_mask].mean().item()
        obj_percent = ((pre_obj[obj_mask] > self.cfg.TEST.SCORE_THRESH).float()).mean().item() if obj_num > 0. else 0.
        noobj_thresh_sum = (pre_obj[noobj_mask] > self.cfg.TEST.SCORE_THRESH).sum().item() / self.batch_size
        cls_p = (pre_cls[obj_mask].max(-1)[1] == lab_cls[obj_mask].max(-1)[1]).float().mean()
        totoal_loss = noobj_loss + obj_loss + cls_loss + loc_loss
        metrics['cls_p'] = cls_p
        metrics['obj_sc'] = obj_sc
        metrics['obj_p'] = obj_percent
        metrics['nob_sc'] = noobj_sc
        metrics['nob_t'] = noobj_thresh_sum
        metrics['ob_l'] = obj_loss.item()
        metrics['nob_l'] = noobj_loss.item()
        metrics['cls_l'] = cls_loss.item()
        metrics['box_l'] = loc_loss.item()
        return totoal_loss, metrics

    def Loss_Call(self, f_maps, dataset, kwargs):
        # timea = time.time()
        images, labels, datainfos = dataset
        metrics = {}
        # I have checked the waste of time ,finally find this line waste a lot of time!!!
        total_loss = torch.FloatTensor([0]).to(self.device)
        # timeb = time.time()
        for f_id, f_map in enumerate(f_maps):
            _loss, _metrics = self._loss_cal_one_Fmap(f_map, f_id, labels, kwargs)
            total_loss += _loss
            if f_id == 0:
                metrics = _metrics
            else:
                for k, v in metrics.items():
                    try:
                        metrics[k] += _metrics[k]
                    except:
                        pass

        for k, v in metrics.items():
            metrics[k] = v / len(f_maps)
        # timec = time.time()
        # print('loss time LOSS CALLb-a:', timeb - timea)
        # print('loss time LOSS CALLc-b:', timec - timeb)
        # print('loss time LOSS CALLc-a:', timec - timea)
        return {'total_loss': total_loss, 'metrics': metrics}
