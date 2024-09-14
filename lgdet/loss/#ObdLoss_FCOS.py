"""Loss calculation based on yolo2."""
import torch
from lgdet.util.util_get_cls_names import _get_class_names


class FCOSLOSS:
    # pylint: disable=too-few-public-methods
    """Calculate loss."""

    def __init__(self, cfg):
        """Init."""
        #
        self.cfg = cfg
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.milabsize = max(self.cfg.TRAIN.IMG_SIZE) / 16 * torch.Tensor([0, 1, 2, 4, 8, 1000])

        self.apollo_cls2idx = dict(zip(cfg.TRAIN.APOLLO_CLASSES, range(len(cfg.TRAIN.APOLLO_CLASSES))))
        self.class_name = _get_class_names(cfg.PATH.CLASSES_PATH)
        # loss fun
        self.mseloss = torch.nn.MSELoss(reduction='sum')
        self.bcelogistloss = torch.nn.BCEWithLogitsLoss()
        self.bceloss = torch.nn.BCELoss()

        # self.MARK used to check whether there is any label that has not been made to be a label.
        self.MARK = torch.zeros(cfg.TRAIN.BATCH_SIZE, 100)
        self.all_not_mark = 0

    def _reshape_labels(self, labels, feature_size, feature_idx):
        """
        Reshape the labels.

        :param labels: labels from training data
        :param grid_xy: the matrix of the grid numbers
        :return: labels_obj, labels_cls, lab_loc_xy, lab_loc_wh, labels_boxes, area_scal
        """
        N = len(labels)
        # print('images', N)
        stride = self.cfg.TRAIN.STRIDES[feature_idx]
        min_size = self.milabsize[feature_idx]
        max_size = self.milabsize[feature_idx + 1]

        # gt_cls = torch.zeros([N, self.cfg.TRAIN.IMG_SIZE[0], self.cfg.TRAIN.IMG_SIZE[1], self.cfg.TRAIN.CLS_NUM])
        # gt_ctness = torch.zeros([N, self.cfg.TRAIN.IMG_SIZE[0], self.cfg.TRAIN.IMG_SIZE[1], 1])
        # gt_loc = torch.zeros([N, self.cfg.TRAIN.IMG_SIZE[0], self.cfg.TRAIN.IMG_SIZE[1], 4])

        gt_cls = torch.zeros([N, feature_size[0], feature_size[1], self.cfg.TRAIN.CLS_NUM])
        gt_ctness = torch.zeros([N, feature_size[0], feature_size[1], 1])
        gt_loc = torch.zeros([N, feature_size[0], feature_size[1], 4])

        # TODO: make a martrix to solve this.
        for bach_n, label in enumerate(labels):
            for i_y in range(feature_size[0]):
                for i_x in range(feature_size[1]):
                    area_min = 100000000
                    for lab_n, lab in enumerate(label):
                        # print('gt label:', lab)
                        cls = lab[0]
                        x1 = lab[1]
                        y1 = lab[2]
                        x2 = lab[3]
                        y2 = lab[4]
                        area = (x2 - x1) * (y2 - y1)

                        l = (i_x * stride + stride / 2 - x1)
                        t = (i_y * stride + stride / 2 - y1)
                        r = (x2 - (i_x * stride + stride / 2))
                        b = (y2 - (i_y * stride + stride / 2))

                        # determine whether the pic is belong to this feature map.
                        if min(l, t, r, b) < 0 or max(l, t, r, b) < min_size or max(l, t, r, b) > max_size:
                            continue
                        # if area is min ,replace the parameters. xy shoud belong to the minimal area of BBOX
                        if area < area_min:
                            area_min = area
                            # cls
                            # print(gt_cls[bach_n, i_y, i_x])
                            gt_cls[bach_n, i_y, i_x] = 0  # clear the max eara cls.
                            gt_cls[bach_n, i_y, i_x, cls] = 1  # fell the min cls.
                            # print(gt_cls[bach_n, i_y, i_x])
                            # loc
                            # l /= self.cfg.TRAIN.IMG_SIZE[1]
                            # t /= self.cfg.TRAIN.IMG_SIZE[0]
                            # r /= self.cfg.TRAIN.IMG_SIZE[1]
                            # b /= self.cfg.TRAIN.IMG_SIZE[0]
                            vector = torch.Tensor([l, t, r, b])
                            gt_loc[bach_n, i_y, i_x] = vector  # replace the max ones directly
                            # cent
                            centerness = torch.sqrt(torch.Tensor([(min(l, r) / max(l, r)) * \
                                                                  (min(t, b) / max(t, b))]))
                            gt_ctness[bach_n, i_y, i_x] = centerness  # replace the max ones directly

                            self.MARK[bach_n, lab_n] = 1

        obj_mask = torch.gt(gt_loc, 0).type(torch.cuda.FloatTensor)
        noobj_mask = 1 - obj_mask
        gt_loc = torch.log(torch.min(torch.max(gt_loc, torch.Tensor([1e-9])
                                               .expand_as(gt_loc)),
                                     torch.Tensor([1e9]).expand_as(gt_loc)))
        return gt_cls.cuda(), gt_ctness.cuda(), gt_loc.cuda(), obj_mask, noobj_mask

    def _reshape_predict(self, predict):
        """
        Reshape the predict, or reshape it to label shape.

        :param predict: out of net
        :param tolabel: True or False to label shape
        :return:
        """
        pre_cls, pre_ctness, pre_loc = predict
        pre_cls, pre_ctness, pre_loc = pre_cls.permute(0, 2, 3, 1), \
                                       pre_ctness.permute(0, 2, 3, 1), \
                                       pre_loc.permute(0, 2, 3, 1)
        # softmax the classes
        clsshape = pre_cls.shape
        feature_size = [clsshape[1], clsshape[2]]
        pre_cls = pre_cls.sigmoid()
        # pre_cls = torch.reshape(pre_cls, (-1, 4)).softmax(-1)
        # pre_cls = torch.reshape(pre_cls, clsshape)
        pre_ctness = pre_ctness.sigmoid()

        return pre_cls, pre_ctness, pre_loc, feature_size

    def loss_cal(self, predicts, labels, losstype=None):
        """Calculate the loss."""
        obj_loss_cls = 0.
        noobj_loss_cls = 0.
        loss_ctness = 0.
        loss_loc = 0.
        for feature_i, predict in enumerate(predicts):
            pre_cls, pre_ctness, pre_loc, feature_size = self._reshape_predict(predict)
            gt_cls, gt_ctness, gt_loc, obj_mask, noobj_mask = self._reshape_labels(labels, feature_size, feature_i)

            # LOSS
            if losstype == 'mse' or losstype is None:
                obj_loss_cls += self.mseloss(pre_cls, gt_cls)
                noobj_loss_cls = 0 * obj_loss_cls
                loss_ctness += self.mseloss(pre_ctness, gt_ctness)
                loss_loc += self.mseloss(pre_loc, gt_loc)
            elif losstype == 'focalloss':
                alpha = 0.7
                # print(torch.max(pre_cls), torch.max(gt_cls))
                pre_obj_cls = obj_mask * pre_cls
                pre_noobj_cls = noobj_mask * pre_cls

                _obj_loss_cls = alpha * pow(torch.ones_like(pre_obj_cls) - pre_obj_cls, 2) * self.bceloss(pre_obj_cls,
                                                                                                          gt_cls * obj_mask)  # focal loss
                obj_loss_cls += (torch.sum(_obj_loss_cls) / self.batch_size)

                _noobj_loss_cls = (1 - alpha) * pow(pre_noobj_cls, 2) * self.bceloss(pre_noobj_cls,
                                                                                     gt_cls * noobj_mask)  # focal loss
                noobj_loss_cls += (torch.sum(_noobj_loss_cls) / self.batch_size)

                _loss_ctness = self.bceloss(pre_ctness * obj_mask, gt_ctness * obj_mask)  # bceloss
                loss_ctness += torch.sum(_loss_ctness) / self.batch_size

                loss_loc += self.mseloss(pre_loc * obj_mask, gt_loc * obj_mask) / self.batch_size

            if feature_i == 2:
                i_0, i_1, i_2 = 0, 13, 19
                print('pre_loc[i_0, i_1, i_2]:', pre_loc[i_0, i_1, i_2])
                print('GT_loc[i_0, i_1, i_2]:', gt_loc[i_0, i_1, i_2])
                print('pre_cls[i_0, i_1, i_2]', pre_cls[i_0, i_1, i_2])
                print('GT_cls[i_0, i_1, i_2]', gt_cls[i_0, i_1, i_2])
                print('pre_ctness', pre_ctness[i_0, i_1, i_2])
                print('gt_ctness', gt_ctness[i_0, i_1, i_2])
                print(pre_cls[0, 0, 0])
                print(gt_cls[0, 0, 0])
                print('obj_mask', obj_mask[i_0, i_1, i_2])
                print('NOobj_mask', noobj_mask[i_0, i_1, i_2])
        # #############
        # used to check whether there is any label that has not been made to be a label.
        ###############
        all_labs = 0
        for n, label in enumerate(labels):
            all_labs += len(label)
        marked_labs = self.MARK.sum().item()
        if all_labs - marked_labs != 0:
            print("=" * 100, all_labs - marked_labs, 'labels ERROR at focs_loss.py ! ')
            self.all_not_mark += (all_labs - marked_labs)
        self.MARK = torch.zeros(self.cfg.TRAIN.BATCH_SIZE, 100)
        print('=' * 10, '\nall_not_mark labels:', self.all_not_mark)
        return obj_loss_cls, noobj_loss_cls, loss_ctness, loss_loc
