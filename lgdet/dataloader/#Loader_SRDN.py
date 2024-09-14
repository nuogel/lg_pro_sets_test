import torch
import numpy as np
import cv2
import random
from lgdet.util.util_data_aug import Dataaug
from lgdet.util.util_JPEG_compression import Jpegcompress2


# import multiprocessing as mp


class Loader:
    def __init__(self, cfg):
        self.cfg = cfg
        self.one_test = cfg.TEST.ONE_TEST
        self.one_name = cfg.TEST.ONE_NAME
        self.train_batch_num = 100
        self.test_batch_num = 1
        self.Data_aug = Dataaug(self.cfg)

    def get_data_by_idx(self, idx_store, index_from, index_to, is_training=True):
        '''
        :param idx_store:
        :param index_from:
        :param index_to:
        :return: imags: torch.Float32, relative labels:[[cls, x1, y1, x2, y2],[...],...]
        '''
        data = (None, None)
        if self.one_test:
            if self.one_name:
                idx = self.one_name
            else:
                idx = idx_store[1:2]
        else:
            idx = idx_store[index_from: index_to]
        if not idx:
            print('error, no IDX in loader_img.py')
            exit()
        # processes = mp.Pool(4)
        # imgs, labels = processes.apply_async(self._prepare_data, (idx))
        imgs, labels = self._prepare_data(idx, is_training)

        imgs = torch.from_numpy(imgs)
        labels = torch.from_numpy(labels)
        imgs = imgs.to(self.cfg.TRAIN.DEVICE)
        labels = labels.to(self.cfg.TRAIN.DEVICE)
        imgs = (imgs - self.cfg.TRAIN.PIXCELS_NORM[0]) / self.cfg.TRAIN.PIXCELS_NORM[1]
        labels = (labels - self.cfg.TRAIN.PIXCELS_NORM[0]) / self.cfg.TRAIN.PIXCELS_NORM[1]
        imgs = imgs.permute([0, 3, 1, 2])
        data = (imgs, labels)  #
        return data

    def _prepare_data(self, idx, is_training=False):
        input_imgs = []
        target_imgs = []
        for id in idx:

            target = self._target_prepare(filename=id)

            input = self._input_prepare(target=target, filename=id, is_training=is_training)

            if self.cfg.TRAIN.SHOW_INPUT:
                cv2.imshow('img', input)
                cv2.waitKey(self.cfg.TRAIN.SHOW_INPUT)

            input_imgs.append(input)
            target_imgs.append(target)

        input_imgs = np.asarray(input_imgs, dtype=np.float32)
        target_imgs = np.asarray(target_imgs, dtype=np.float32)

        return input_imgs, target_imgs

    def _target_prepare(self, **kwargs):

        id = kwargs['filename']
        target = cv2.imread(id[2])  # no norse image or HR image
        if target is None:
            print(id, 'image is wrong!!')
            exit()
        if self.cfg.TRAIN.TARGET_PREDEAL:
            # add the pre deal programs.
            # target = _crop_licience_plante(target, id[0])
            target, _ = self.Data_aug.augmentation(for_one_image=[target])
            target = target[0]
            pass

        target = cv2.resize(target, (self.cfg.TRAIN.IMG_SIZE[0], self.cfg.TRAIN.IMG_SIZE[1]))

        return target

    def _input_prepare(self, **kwargs):
        target = kwargs['target']
        id = kwargs['filename']
        is_training = kwargs['is_training']
        input_is_target = True  # id[1] in ['None', '', ' ', 'none']

        if input_is_target:
            input = target
        else:
            input = cv2.imread(id[1])

        if self.cfg.TRAIN.MODEL not in ['cbdnet', 'dbpn', 'dncnn']:  # 去噪网络就不用缩小尺寸
            input = cv2.resize(input, (self.cfg.TRAIN.IMG_SIZE[0] // self.cfg.TRAIN.UPSCALE_FACTOR,  # SR model 使用
                                       self.cfg.TRAIN.IMG_SIZE[1] // self.cfg.TRAIN.UPSCALE_FACTOR))

        if self.cfg.TRAIN.INPUT_AUG:# and is_training:
            # add the augmentation ...
            input = np.asarray(input, np.float32)
            input, _ = self.Data_aug.augmentation(for_one_image=[input])
            input = input[0]
            compress_level = random.randint(5, 20)
            input = Jpegcompress2(input, compress_level)

        if self.cfg.TRAIN.MODEL in ['cbdnet', 'srcnn', 'vdsr']:
            input = cv2.resize(input, (self.cfg.TRAIN.IMG_SIZE[0], self.cfg.TRAIN.IMG_SIZE[1]))

        return input
