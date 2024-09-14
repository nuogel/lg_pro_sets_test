"""Parse the predictions."""
import torch
import numpy as np
from lgdet.util.util_nms.util_nms_python import NMS
import torchvision


class ParsePredict_yolox:
    # TODO: 分解parseprdict.
    def __init__(self, cfg):
        self.cfg = cfg
        self.anchors = torch.Tensor(cfg.TRAIN.ANCHORS[::-1])  # SMALL -> BIG
        self.anc_num = cfg.TRAIN.FMAP_ANCHOR_NUM
        self.cls_num = cfg.TRAIN.CLASSES_NUM
        self.device = self.cfg.TRAIN.DEVICE
        self.decode_in_inference = 1
        self.strides = [8, 16, 32]
        self.confthre = 0.5
        self.nmsthre = cfg.TEST.IOU_THRESH

        if self.cfg.TEST.MAP_FSCORE in [0, '0']:  # 1-fscore
            self.confthre = 0.05  # count map score thresh is <0.05

    def parse_predict(self, f_maps):
        """
        Parse the predict. with all feature maps to labels.

        :param f_maps: predictions out of net.
        :return:parsed predictions, to labels.
        """
        self.hw = [x.shape[-2:] for x in f_maps]
        # [batch, n_anchors_all, 85]
        outputs = torch.cat([x.flatten(start_dim=2) for x in f_maps], dim=2).permute(0, 2, 1)
        outputs[..., 4:] = outputs[..., 4:].sigmoid()
        if self.decode_in_inference:
            outputs = self.decode_outputs(outputs, dtype=f_maps[0].type())
            outputs = self.postprocess(outputs, self.cls_num, self.confthre, self.nmsthre)
        return outputs

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs

    def postprocess(self, prediction, num_classes, conf_thre=0.7, nms_thre=0.45):
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [[] for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):

            # If none are remaining => process next image
            if not image_pred.size(0):
                continue
            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

            conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
            # _, conf_mask = torch.topk((image_pred[:, 4] * class_conf.squeeze()), 1000)
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            # detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
            detections = torch.cat((class_conf * image_pred[:, 4:5], class_pred.float(), image_pred[:, :4]), 1)
            detections = detections[conf_mask]
            if not detections.size(0):
                continue

            nms_out_index = torchvision.ops.batched_nms(
                detections[:, 2:],
                detections[:, 0],
                detections[:, 1],
                nms_thre, )
            detections = detections[nms_out_index]
            if output[i] == []:
                output[i] = detections
            else:
                output[i] = torch.cat((output[i], detections))

        return output
