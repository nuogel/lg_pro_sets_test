import os
import torch
import random
import cv2
from lgdet.util.util_data_aug import Dataaug
import numpy as np
import xml.etree.ElementTree as ET
from lgdet.util.util_get_cls_names import _get_class_names
from lgdet.util.util_show_img import _show_img
from torch.utils.data import DataLoader


class Loader(DataLoader):
    def __init__(self, cfg):
        super(Loader, self).__init__(object)
        self.cfg = cfg
        self.dataaug = Dataaug(cfg)
        self.one_test = cfg.TEST.ONE_TEST
        self.one_name = cfg.TEST.ONE_NAME
        self.train_batch_num = 100
        self.test_batch_num = 1
        self.print_path = self.cfg.TRAIN.SHOW_TRAIN_NAMES

    def __len__(self):
        if self.one_test:
            if self.is_training:
                length = int(self.cfg.TEST.ONE_TEST_TRAIN_STEP)
            else:
                length = len(self.cfg.TEST.ONE_NAME)
        else:
            length = len(self.dataset_txt)
        return length

    def __getitem__(self, index):
        if self.one_test:
            data_info = self.dataset_txt[0]
        else:
            data_info = self.dataset_txt[index]
        a=0

    def _add_dataset(self, dataset, is_training):
        self.dataset_txt = dataset
        self.is_training = is_training

    def collate_fun(self, batch):
        ...