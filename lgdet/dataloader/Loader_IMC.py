import os
import torch
import random
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from lgdet.util.util_get_cls_names import _get_class_names
from lgdet.util.util_show_img import _show_img
from torch.utils.data import DataLoader
from ..registry import DATALOADERS
from lgdet.util.util_lg_transformer import LgTransformer
import pickle
import tqdm
import math


@DATALOADERS.registry()
class IMC_Loader(DataLoader):
    def __init__(self, cfg, dataset, is_training):
        super(IMC_Loader, self).__init__(object)
        self.cfg = cfg
        self.is_training = is_training
        self.one_test = cfg.TEST.ONE_TEST
        self.one_name = cfg.TEST.ONE_NAME
        self.class_name = _get_class_names(cfg.PATH.CLASSES_PATH)
        self.print_path = self.cfg.TRAIN.SHOW_TRAIN_NAMES
        self.cls2idx = dict(zip(cfg.TRAIN.CLASSES, range(cfg.TRAIN.CLASSES_NUM)))
        self.write_images = self.cfg.TRAIN.WRITE_IMAGES
        self.lgtransformer = LgTransformer(self.cfg)
        self.keep_difficult = True
        if self.cfg.TRAIN.MULTI_SCALE:
            self._prepare_multiszie()
        if self.cfg.TRAIN.USE_LMDB:
            import lmdb
            mod = 'train' if self.is_training else 'val'
            lmdb_path = self.cfg.PATH.LMDB_PATH.format(mod)
            env = lmdb.open(lmdb_path, max_dbs=2, lock=False)
            db_image = env.open_db('image'.encode())
            db_label = env.open_db('label'.encode())
            self.txn_image = env.begin(write=False, db=db_image)
            self.txn_label = env.begin(write=False, db=db_label)
        if is_training is None:
            pre_load_labels = 0
        else:
            pre_load_labels = 1
            print('pre-loading labels to disc...')
        if self.one_test:
            self.dataset_infos = self.cfg.TEST.ONE_NAME
        else:
            self.dataset_infos = dataset

    def __len__(self):
        if self.one_test:
            if self.is_training:
                length = int(self.cfg.TEST.ONE_TEST_TRAIN_STEP) * self.cfg.TRAIN.BATCH_SIZE
            else:
                length = self.cfg.TRAIN.BATCH_SIZE
        else:
            length = len(self.dataset_infos)
            # length = 24
        return length

    def __getitem__(self, index):
        img, label, data_info = self._load_gt(index)

        # GRAY_BINARY
        if (self.cfg.TRAIN.GRAY_BINARY and self.is_training) or (self.cfg.TEST.GRAY_BINARY and not self.is_training):
            img, _, data_info = self.lgtransformer.img_binary(img, None, data_info)

        # DOAUG:
        if (random.random() < self.cfg.TRAIN.DO_AUG) and self.is_training:  # data aug is wasting time.
            img, data_info = self.lgtransformer.data_aug_classification(img, data_info)

        # augment_hsv
        if (random.random() < self.cfg.TRAIN.HSV) and self.is_training:  # data aug is wasting time.
            img, _, data_info = self.lgtransformer.augment_hsv(img, None, data_info)

        # resize with max and min size ([800, 1333])
        if (self.cfg.TRAIN.MAXMINSIZE and self.is_training) or (self.cfg.TEST.MAXMINSIZE and not self.is_training):
            img, _, data_info = self.lgtransformer.resize_max_min_size(img, None, data_info, input_ksize=self.cfg.TRAIN.IMG_SIZE)

        # resize
        if (self.cfg.TRAIN.RESIZE and self.is_training) or (self.cfg.TEST.RESIZE and not self.is_training):
            img = cv2.resize(img, (self.cfg.TRAIN.IMG_SIZE[1], self.cfg.TRAIN.IMG_SIZE[0]), interpolation=cv2.INTER_CUBIC)

        # PAD TO SIZE:
        if (self.cfg.TRAIN.LETTERBOX and self.is_training) or (self.cfg.TEST.LETTERBOX and not self.is_training):
            img, _, data_info = self.lgtransformer.letter_box(img, [], data_info, new_shape=self.cfg.TRAIN.IMG_SIZE, auto=False, scaleup=True)

        if self.cfg.TRAIN.SHOW_INPUT > 0:
            _show_img(img.copy(),[], cfg=self.cfg, show_time=self.cfg.TRAIN.SHOW_INPUT)

        if self.write_images > 0 and self.is_training and not self.cfg.checkpoint:
            # img_write = _show_img(img.copy(), label.copy(), cfg=self.cfg, show_time=-1)[0]
            self.cfg.writer.tbX_addImage('label:'+str(label), img)
            self.write_images -= 1

        img, _ = self.lgtransformer.transpose(img)
        assert len(img) > 0, 'img length is error'
        label = self.cls2idx[label]
        return [img, label, data_info]  # only need the labels  label_after[x1y1x2y2]

    def _set_group_flag(self):  # LG: this is important
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.data_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _load_gt(self, index):
        '''
        Xxxx
        :param index:
        :return: img, list(label)
        '''
        img = None
        label = None
        data_info={}
        while img is None or label is None:  # if there is no data in img or label
            if self.one_test:
                _data_info = self.dataset_infos[0]
            else:
                _data_info = self.dataset_infos[index]
            img = cv2.imread(_data_info[1])
            label = _data_info[2]
            data_info = {'img':_data_info[1], 'label':_data_info[2]}
        return img, label, data_info

    def _add_dataset(self, dataset, is_training):
        self.dataset_txt = dataset
        self.is_training = is_training

    def _prepare_multiszie(self):
        self.gs = 32
        imgsz_min, imgsz_max = self.cfg.TRAIN.MULTI_SCALE_SIZE
        self.grid_min, self.grid_max = imgsz_min // self.gs, imgsz_max // self.gs
        self.imgsz_min, self.imgsz_max = int(self.grid_min * self.gs), int(self.grid_max * self.gs)
        self.img_size = imgsz_max  # initialize with max size

    def collate_fun(self, batch):
        '''
        collate_fn：如何取样本的，我们可以定义自己的函数来准确地实现想要的功能
        其中default_collate会将labels分割合并转换成tensor。
        !!!***if not use my own collect_fun ,the labels will be wrong orders.***
        :param batch:
        :return:
        '''
        imgs, labels, infos = zip(*batch)

        h_list = [int(s.shape[1]) for s in imgs]
        w_list = [int(s.shape[2]) for s in imgs]
        max_h = np.array(h_list).max()
        max_w = np.array(w_list).max()
        pad_imgs = []
        for i in range(len(imgs)):
            img = imgs[i]
            pad_imgs.append(torch.nn.functional.pad(img, (0, int(max_w - img.shape[2]), 0, int(max_h - img.shape[1])), value=0.))

        imgs = torch.stack(pad_imgs, dim=0)
        if self.cfg.TRAIN.MULTI_SCALE and self.is_training:
            img_size = self.img_size
            if random.random() > 0.5:  #  adjust img_size (67% - 150%) every 1 batch
                img_size = random.randrange(self.grid_min, self.grid_max + 1) * self.gs
            sf = img_size / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [math.ceil(x * sf / self.gs) * self.gs for x in imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                imgs = torch.nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

        labels = torch.LongTensor(list(labels))
        return [imgs, labels, list(infos)]
