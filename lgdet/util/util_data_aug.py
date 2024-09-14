"""Collection of data augumentation functions."""
import random

import cv2
import imgaug as ia
import numpy as np
from imgaug import augmenters as iaa
import xml.etree.ElementTree as ET


class Dataaug:
    '''
    Data augmentation.
    '''

    def __init__(self, cfg=None):
        self.cfg = cfg

    def augmentation(self, aug_way_ids, datas):
        images, labels = self._augmenting(aug_way_ids, datas)
        # images, labels = self._check_results(images, labels)
        return images, labels

    def _augmenting(self, aug_way_ids, datas):
        '''

        :param aug_way_ids: [[random ids], [must ids]]
        :param datas:
        :return:
        '''

        if self.cfg:
            crop_size = self.cfg.TRAIN.IMG_SIZE
        else:
            crop_size = (512, 512)
        base_funs = {
            ##############################################
            ########     color augmentation   #############

            3: iaa.Grayscale(alpha=(0, 1)),
            4: iaa.ChangeColorspace('BGR'),
            5: iaa.Add((-50, 50)),
            6: iaa.Dropout(0.08, per_channel=0.5),
            7: iaa.GammaContrast(gamma=(0.5, 1.5), per_channel=True),
            ## 设置 加 雨 的噪声 类型
            8: iaa.Snowflakes(density=(0, 0.15), density_uniformity=0.08, flake_size=0.4, flake_size_uniformity=0.5, speed=0.1),
            # 加 雪花 的噪声 类型
            9: iaa.Snowflakes(density=(0, 0.15), density_uniformity=0.5, flake_size=0.4, flake_size_uniformity=0.5, speed=0.001),
            #### 运动模糊
            10: iaa.MotionBlur(k=(3, 11), angle=[-45, 45]),
            # 高斯模糊
            11: iaa.GaussianBlur((0, 3)),
            # 高斯噪音
            12: iaa.AdditiveGaussianNoise(scale=(0, 0.15 * 255)),
            # 拉普拉斯
            13: iaa.AdditiveLaplaceNoise(scale=(0, 0.15 * 255)),
            # 泊松
            14: iaa.AdditivePoissonNoise(lam=(0.0, 60.0)),
            # SaltAndPepper
            15: iaa.Salt(0.1),
            # 椒盐
            16: iaa.SaltAndPepper((0.03, 0.1)),
            17: iaa.Fog(),
            18: iaa.Clouds(),
            ####################
            #####   shape aug  >20 ###
            # 19: iaa.Fliplr(0.5),  # 增加原图的概率。
            20: iaa.Fliplr(0.5),  # 增加原图的概率。
            # 21: iaa.Fliplr(0.5),  # 增加原图的概率。
            22: iaa.Affine(translate_percent=(-0.01, 0.01), rotate=(-3, 3)),
            23: iaa.Crop(),
            24: iaa.CropAndPad(),
            25: iaa.CropToFixedSize(width=crop_size[1], height=crop_size[0], position='normal'),
            26: iaa.PadToFixedSize(width=crop_size[1], height=crop_size[0], ),
            27: iaa.Resize(crop_size)
        }

        _base_funs = [base_funs.get(id) for id in aug_way_ids[0]]
        must_funs = [base_funs.get(id) for id in aug_way_ids[1]]
        aug_funs = [random.choice(_base_funs)] + must_funs

        # do the augmentation
        seq_det = iaa.Sequential(aug_funs)
        seq_det = seq_det.to_deterministic()

        images, labels = datas
        if labels not in [None, 0, 'None']:
            labels = [[ia.BoundingBox(x1=labs[1], y1=labs[2], x2=labs[3], y2=labs[4], label=labs[0]) for labs in _labels] for _labels in labels]
            labels = [ia.BoundingBoxesOnImage(x, shape=images[i].shape) for i, x in enumerate(labels)]
            bbs_aug = seq_det.augment_bounding_boxes(labels)
            labels = [self._parse_bbs(x) for x in bbs_aug]
        images = seq_det.augment_images(images)

        return images, labels

    def _parse_bbs(self, bbs_aug):
        """
        Parse the key points from augmentation.

        :param bbs_aug: key points from augmentation.
        :param relative: the relative location of bounding boxes
        :return:labs are in the shape of [[class, left, top, button, right],..]
        """
        labs = []
        height = bbs_aug.height
        width = bbs_aug.width
        for bb_aug in bbs_aug.bounding_boxes:
            _bb_x1, _bb_y1, _bb_x2, _bb_y2, cls = bb_aug.x1, bb_aug.y1, bb_aug.x2, bb_aug.y2, bb_aug.label
            bb_x1 = min(max(1, _bb_x1), width - 1)
            bb_y1 = min(max(1, _bb_y1), height - 1)
            bb_x2 = min(max(1, _bb_x2), width - 1)
            bb_y2 = min(max(1, _bb_y2), height - 1)
            # calculate the ratio of area
            area_before = (_bb_x2 - _bb_x1) * (_bb_y2 - _bb_y1)
            area_after = (bb_x2 - bb_x1) * (bb_y2 - bb_y1)
            if area_after / area_before < self.cfg.TRAIN.AREAR_RATIO or area_after < self.cfg.TRAIN.MIN_AREA:
                continue
            # get the lab
            lab = [cls, bb_x1, bb_y1, bb_x2, bb_y2]
            labs.append(lab)
        return labs

    def _check_results(self, images, labels):
        if labels in [None, 0, 'None']:
            pass
        else:
            # if labels is not none ,then check the label in labels,whether the label is none.
            for i, label in enumerate(labels):
                j = 1
                while not label:  # check every label, whether there is a label is empty.
                    # print('When trying augmentation, no label at NO.', i)
                    try:
                        label = labels[i - j]
                        labels[i] = labels[i - j]
                        images[i] = images[i - j]
                        j += 1
                    except:
                        return images, 'None'
        return images, labels


if __name__ == '__main__':
    aug = Dataaug()
    img_path = 'E:/datasets/kitti/training/images/000076.png'
    img = cv2.imread(img_path)
    imgs, lables = aug.augmentation(aug_way_ids=[[8], []], datas=([img], None))
    cv2.imwrite('aug.png', imgs[0])
    cv2.imshow('img', imgs[0])
    cv2.waitKey()
