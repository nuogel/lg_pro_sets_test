import torch
import cv2
from lgdet.util.util_data_aug import Dataaug
import numpy as np
from lgdet.util.util_get_cls_names import _get_class_names
from torch.utils.data import DataLoader
from lgdet.util.util_show_FLOW import _viz_flow
from ..registry import DATALOADERS


@DATALOADERS.registry()
class FLOW_Loader(DataLoader):
    def __init__(self, cfg):
        super(FLOW_Loader, self).__init__(object)
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

        if self.one_test:
            data_info = self.dataset_txt[0]
        else:
            data_info = self.dataset_txt[index]

        img, label = self._read_datas(data_info)

        if self.cfg.TRAIN.SHOW_INPUT:
            _viz_flow(img, label)
        img = np.concatenate(img, axis=-1)
        img = np.asarray(img, dtype=np.float32)
        img = np.transpose(img, (2, 0, 1))
        img = img / 127.5 - 1.

        label = np.transpose(label, (2, 0, 1))

        return img, label, data_info  # only need the labels

    def _add_dataset(self, dataset, is_training):
        self.dataset_txt = dataset
        self.is_training = is_training

    def _load_flo(self, path):
        with open(path, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)
            assert (202021.25 == magic), 'Magic number incorrect. Invalid .flo file'
            h = np.fromfile(f, np.int32, count=1)[0]
            w = np.fromfile(f, np.int32, count=1)[0]
            data = np.fromfile(f, np.float32, count=2 * w * h)
        # Reshape data into 3D array (columns, rows, bands)
        data2D = np.resize(data, (w, h, 2))
        return data2D

    def _read_datas(self, datainfo):
        img1 = cv2.imread(datainfo[0])
        img2 = cv2.imread(datainfo[1])
        label = self._load_flo(datainfo[2])
        return [img1, img2], label

    def collate_fun(self, batch):
        '''
        collate_fn：如何取样本的，我们可以定义自己的函数来准确地实现想要的功能
        其中default_collate会将labels分割合并转换成tensor。
        !!!***if not use my own collect_fun ,the labels will be wrong orders.***
        :param batch:
        :return:
        '''
        imgs, labels, infos = zip(*batch)
        imgs = torch.from_numpy(np.asarray(imgs))
        labels = torch.from_numpy(np.asarray(labels))
        return imgs, labels, list(infos)


