import os
import torch
import random
import cv2
from lgdet.util.util_data_aug import Dataaug
import numpy as np
import xml.etree.ElementTree as ET
from lgdet.util.util_get_cls_names import _get_class_names
from lgdet.util.util_show_img import _show_img


class Loader:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataaug = Dataaug(cfg)
        self.one_test = cfg.TEST.ONE_TEST
        self.one_name = cfg.TEST.ONE_NAME
        self.train_batch_num = 100
        self.test_batch_num = 1
        self.class_name = _get_class_names(cfg.PATH.CLASSES_PATH)
        self.print_path = self.cfg.TRAIN.SHOW_TRAIN_NAMES
        self.cls2idx = dict(zip(self.cfg.TRAIN.CLASSES, range(len(self.cfg.TRAIN.CLASSES))))
        self.a = 0

    def get_data_by_idx(self, idx_store, index_from, index_to, is_training):
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
        imgs, labels = self._read_datas(idx)
        do_aug = self.cfg.TRAIN.DO_AUG if is_training else self.cfg.TEST.DO_AUG
        if do_aug:
            imgs, labels = self.dataaug.augmentation((imgs, labels))
        if self.cfg.TRAIN.RESIZE:
            size = random.choice(self.cfg.TRAIN.MULTI_SIZE_RATIO) * self.cfg.TRAIN.IMG_SIZE
            resized_imgs = []
            resized_labels = []
            for i, img_i in enumerate(imgs):
                img_i_size = img_i.shape
                resized_imgs.append(cv2.resize(img_i, (size[1], size[0])))
                labs_i = labels[i]
                if self.cfg.TRAIN.RELATIVE_LABELS:
                    label_i = [[lab[0],
                                lab[1] / img_i_size[1],
                                lab[2] / img_i_size[0],
                                lab[3] / img_i_size[1],
                                lab[4] / img_i_size[0]
                                ] for lab in labs_i]
                else:
                    label_i = [[lab[0],
                                lab[1] / img_i_size[1] * size[1],
                                lab[2] / img_i_size[0] * size[0],
                                lab[3] / img_i_size[1] * size[1],
                                lab[4] / img_i_size[0] * size[0]
                                ] for lab in labs_i]
                resized_labels.append(label_i)
            imgs = resized_imgs
            labels = resized_labels

            if self.cfg.TRAIN.SHOW_INPUT:
                _show_img(imgs, labels, show_time=self.cfg.TRAIN.SHOW_INPUT, cfg=self.cfg)

            imgs = torch.Tensor(np.array(imgs))
            imgs = imgs.permute([0, 3, 1, 2, ])
            imgs = imgs / 127.5 - 1.

            imgs = imgs.to(self.cfg.TRAIN.DEVICE)
            data = (imgs, labels)  #
        return data

    def _read_line(self, path, predicted_line=False, pass_obj=['DontCare', ]):
        """
        Parse the labels from file.

        :param pass_obj: pass the labels in the list.
                        e.g. pass_obj=['Others','Pedestrian', 'DontCare']
        :param path: the path of file that need to parse.
        :return:lists of the classes and the key points.
        """
        bbs = []
        if os.path.basename(path).split('.')[-1] == 'txt' and not predicted_line:
            file_open = open(path, 'r')
            for line in file_open.readlines():
                tmps = line.strip().split(' ')
                if tmps[0] not in self.class_name:
                    continue
                if self.class_name[tmps[0]] in pass_obj:
                    continue
                box_x1 = float(tmps[4])
                box_y1 = float(tmps[5])
                box_x2 = float(tmps[6])
                box_y2 = float(tmps[7])
                if not self._is_finedata([box_x1, box_y1, box_x2, box_y2]): continue
                bbs.append([self.cls2idx[self.class_name[tmps[0]]], box_x1, box_y1, box_x2, box_y2])
        elif os.path.basename(path).split('.')[-1] == 'xml' and not predicted_line:
            tree = ET.parse(path)
            root = tree.getroot()
            for obj in root.findall('object'):
                cls_name = obj.find('name').text
                if cls_name not in self.class_name:
                    continue
                if self.class_name[cls_name] in pass_obj:
                    continue
                bbox = obj.find('bndbox')
                box_x1 = float(bbox.find('xmin').text)
                box_y1 = float(bbox.find('ymin').text)
                box_x2 = float(bbox.find('xmax').text)
                box_y2 = float(bbox.find('ymax').text)
                if not self._is_finedata([box_x1, box_y1, box_x2, box_y2]): continue
                bbs.append([self.cls2idx[self.class_name[cls_name]], box_x1, box_y1, box_x2, box_y2])

        elif os.path.basename(path).split('.')[-1] == 'txt' and predicted_line:
            f_path = open(path, 'r')
            print(path)
            for line in f_path.readlines():
                tmp = line.split()
                cls_name = tmp[0]
                bbs.append([float(tmp[1]), self.cls2idx[self.class_name[cls_name]], [float(tmp[2]), float(tmp[3]), float(tmp[4]), float(tmp[5])]])
        return bbs

    def _read_datas(self, idx, image=None):
        '''
        Read images and labels depend on the idx.
        :param image: if there is a image ,the just return the image.
        :return: images, labels, image_size
        '''
        images = []
        labels = []
        if image is None:
            idx_remove = idx.copy()
            for id in idx:
                if self.print_path:
                    print(id[1], '<--->', id[2])
                if not (os.path.isfile(id[1]) and os.path.isfile(id[2])):
                    print('ERROR, NO SUCH A FILE.', id[1], '<--->', id[2])
                    exit()
                # labels come first.
                label = self._read_line(id[2])
                while not label:  # whether the label is empty?
                    if id in idx_remove:  # if img_idx has been removed then label is empty
                        idx_remove.remove(id)
                    if not idx_remove: break
                    id = random.choice(idx_remove)
                    print('warning: no label...instead it of ', id[1])
                    label = self._read_line(id[2])
                labels.append(label)
                # then add the images.
                img = cv2.imread(id[1])
                if img is not None:
                    images.append(img)
                else:
                    print('imread img is NONE.')
        else:
            images.append(image)

        if labels == [[]]:
            labels = None
        return images, labels

    def _is_finedata(self, xyxy):
        x1, y1, x2, y2 = xyxy
        for point in xyxy:
            if point < 0: return False
        if x2 - x1 <= 0: return False
        if y2 - y1 <= 0: return False
        return True
