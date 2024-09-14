import torch
import random
import cv2
from lgdet.util.util_data_aug import Dataaug
import numpy as np
from lgdet.util.util_get_cls_names import _get_class_names
from lgdet.util.util_show_img import _show_img
from torch.utils.data import DataLoader
from lgdet.util.util_make_VisDrone2019_VID_dataset import make_VisDrone2019_VID_dataset

from ..registry import DATALOADERS


@DATALOADERS.registry()
class VID_Loader(DataLoader):
    def __init__(self, cfg):
        super(VID_Loader, self).__init__(object)
        self.cfg = cfg
        self.dataaug = Dataaug(cfg)
        self.one_test = cfg.TEST.ONE_TEST
        self.one_name = cfg.TEST.ONE_NAME
        self.class_name = _get_class_names(cfg.PATH.CLASSES_PATH)
        self.print_path = self.cfg.TRAIN.SHOW_TRAIN_NAMES
        self.cls2idx = dict(zip(self.cfg.TRAIN.CLASSES, range(len(self.cfg.TRAIN.CLASSES))))
        self.a = 0
        self.vid_sequeece = None
        self.vid_data = None
        self.refnum = 4  # [-4, +4]

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
        img = None
        label = None
        # if self.is_training:index = 4

        while img is None or label is None:  # if there is no data in img or label
            if self.one_test:
                data_info = self.dataset_txt[0]
            else:
                data_info = self.dataset_txt[index]

            if self.vid_sequeece == data_info[2]:
                vid_data = self.vid_data
            else:
                vid_data = make_VisDrone2019_VID_dataset(path='E:/datasets/VisDrone2019/VisDrone2019-VID-train/')
                self.vid_sequeece = data_info[2]
                self.vid_data = vid_data

            img, imgs_ref, label = self._read_datas(vid_data[0], index)
            if not self.is_training and not label:
                break
            index += 1

        if self.cfg.TRAIN.DO_AUG and self.is_training:
            labels = 'None'
            try_tims = 0
            while labels is 'None':
                imgs, labels = self.dataaug.augmentation(aug_way_ids=([20, 22], [25]), datas=([img], [label]))
                try_tims += 1
                if try_tims > 100:
                    print('trying', try_tims, ' times when data augmentation at file:', str(data_info[2]))
            img = imgs[0]
            label = labels[0]
        img_i_size = img.shape
        size = img_i_size

        if (self.cfg.TRAIN.RESIZE and self.is_training) or (self.cfg.TEST.RESIZE and not self.is_training):
            size = random.choice(self.cfg.TRAIN.MULTI_SIZE_RATIO) * self.cfg.TRAIN.IMG_SIZE
            img_out = []
            img_out.append(cv2.resize(img, (size[1], size[0])))
            for img_i in imgs_ref:
                img_out.append(cv2.resize(img_i, (size[1], size[0])))
        if not label:
            label_after = None
        elif self.cfg.TRAIN.RELATIVE_LABELS:  # x1y1x2y2
            label_after = [[0,
                            lab[0],
                            lab[1] / img_i_size[1],
                            lab[2] / img_i_size[0],
                            lab[3] / img_i_size[1],
                            lab[4] / img_i_size[0]
                            ] for lab in label]
        else:
            label_after = [[0,
                            lab[0],
                            lab[1] / img_i_size[1] * size[1],
                            lab[2] / img_i_size[0] * size[0],
                            lab[3] / img_i_size[1] * size[1],
                            lab[4] / img_i_size[0] * size[0]
                            ] for lab in label]

        if self.cfg.TRAIN.SHOW_INPUT:
            _show_img(img, label_after, cfg=self.cfg, show_time=self.cfg.TRAIN.SHOW_INPUT, pic_path=data_info[2])

        img_out = np.asarray(img_out, dtype=np.float32)
        img_out = np.transpose(img_out, (0, 3, 1, 2))
        img_out = img_out / 127.5 - 1.
        if label_after: label_after = torch.Tensor(label_after)
        return img_out, label_after, data_info  # only need the labels

    def _add_dataset(self, dataset, is_training):
        self.dataset_txt = dataset
        self.is_training = is_training

    def _read_datas(self, vid_data, index):
        len = vid_data.__len__()
        if 0 < len < self.refnum:
            id_left = 0
            id_right = len
        elif index - self.refnum / 2 < 0:
            id_left = 0
            id_right = self.refnum
        elif index + self.refnum / 2 > len-2:
            id_left = index - self.refnum
            id_right = len-2
        else:
            id_left = index - int(self.refnum/2)
            id_right =index + int(self.refnum/2)

        image_bef_ids = [i for i in range(id_left + 1, index + 1)]
        image_aft_ids = [i for i in range(index + 1 + 1, id_right + 1 + 1)]

        label = vid_data[index + 1]
        img_path = label[0]
        img = cv2.imread(img_path)
        ref_image_ids = image_bef_ids + image_aft_ids
        imgs_ref = []
        for id in ref_image_ids:
            imgs_ref.append(cv2.imread(vid_data[id][0]))

        label = label[1:]
        return img, imgs_ref, label

    def _is_finedata(self, xyxy):
        x1, y1, x2, y2 = xyxy
        for point in xyxy:
            if point <= 0.: return False
        if x2 - x1 <= 0.: return False
        if y2 - y1 <= 0.: return False
        return True

    def collate_fun(self, batch):
        '''
        collate_fn：如何取样本的，我们可以定义自己的函数来准确地实现想要的功能
        其中default_collate会将labels分割合并转换成tensor。
        !!!***if not use my own collect_fun ,the labels will be wrong orders.***
        :param batch:
        :return:
        '''
        imgs, labels, infos = zip(*batch)
        imgs = torch.from_numpy(np.asarray(imgs[0], np.float32))

        return imgs, labels[0], list(infos)
