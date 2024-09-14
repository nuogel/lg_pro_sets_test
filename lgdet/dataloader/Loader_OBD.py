#!/usr/bin/env Python
# coding=utf-8

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
class OBD_Loader(DataLoader):
    def __init__(self, cfg, dataset, is_training):
        super(OBD_Loader, self).__init__(object)
        self.cfg = cfg
        self.is_training = is_training
        self.one_test = cfg.TEST.ONE_TEST
        self.one_name = cfg.TEST.ONE_NAME
        self.class_name = _get_class_names(cfg.PATH.CLASSES_PATH)
        self.print_path = self.cfg.TRAIN.SHOW_TRAIN_NAMES
        self.cls2idx = dict(zip(cfg.TRAIN.CLASSES, range(cfg.TRAIN.CLASSES_NUM)))
        self.write_images = self.cfg.TRAIN.WRITE_IMAGES
        self.lgtransformer = LgTransformer(self.cfg)
        self.keep_difficult = self.cfg.TRAIN.KEEP_DIFFICAULT
        self.passedlabels = []
        self.wronglabel = []

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
        self.dataset_infos = self._load_labels2memery(dataset, self.one_name, pre_load_labels)
        # self.dataset_infos=self.dataset_infos[:10000]  # 当数量过大时，训练速度变慢, 最后发现是数据增强部分对该部分也有影响。

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
        # TODO: discard the small target.
        # GRAY_BINARY
        if (self.cfg.TRAIN.GRAY_BINARY and self.is_training) or (self.cfg.TEST.GRAY_BINARY and not self.is_training):
            img, label, data_info = self.lgtransformer.img_binary(img, label, data_info)
        # MOSAIC
        if (random.random() < self.cfg.TRAIN.MOSAIC) and self.is_training:
            # need 4 images
            indices = [random.randint(0, len(self.dataset_infos) - 1) for _ in range(3)]  # 3 additional image indices
            imglabs = [self._load_gt(index) for index in indices]
            imglabs.insert(0, (img, label, data_info))
            img, label = self.lgtransformer.aug_mosaic(imglabs, s=self.cfg.TRAIN.IMG_SIZE[0])
            while len(label) <= 0:  # if there is no label in mosaic, then repeat mosaic
                img, label = self.lgtransformer.aug_mosaic(imglabs, s=self.cfg.TRAIN.IMG_SIZE[0])
        # DOAUG:
        if (random.random() < self.cfg.TRAIN.DO_AUG) and self.is_training:  # data aug is wasting time.
            img, label, data_info = self.lgtransformer.data_aug(img, label, data_info)

        # augment_hsv
        if (random.random() < self.cfg.TRAIN.HSV) and self.is_training:  # data aug is wasting time.
            img, label, data_info = self.lgtransformer.augment_hsv(img, label, data_info)

        # PAD TO SIZE:
        if (self.cfg.TRAIN.LETTERBOX and self.is_training) or (self.cfg.TEST.LETTERBOX and not self.is_training):
            img, label, data_info = self.lgtransformer.letter_box(img, label, data_info, new_shape=self.cfg.TRAIN.IMG_SIZE, auto=False, scaleup=True)

        # resize with max and min size ([800, 1333])
        if (self.cfg.TRAIN.MAXMINSIZE and self.is_training) or (self.cfg.TEST.MAXMINSIZE and not self.is_training):
            img, label, data_info = self.lgtransformer.resize_max_min_size(img, label, data_info, input_ksize=self.cfg.TRAIN.IMG_SIZE)

        # resize
        if (self.cfg.TRAIN.RESIZE and self.is_training) or (self.cfg.TEST.RESIZE and not self.is_training):
            img, label, data_info = self.lgtransformer.resize(img, label, self.cfg.TRAIN.IMG_SIZE, data_info)

        if self.cfg.TRAIN.RELATIVE_LABELS:
            img, label = self.lgtransformer.relative_label(img, label, )

        if self.cfg.TRAIN.SHOW_INPUT > 0:
            _show_img(img.copy(), label.copy(), cfg=self.cfg, show_time=self.cfg.TRAIN.SHOW_INPUT, pic_path=data_info['lab_path'])

        if self.write_images > 0 and self.is_training and not self.cfg.checkpoint:
            img_write = _show_img(img.copy(), label.copy(), cfg=self.cfg, show_time=-1)[0]
            self.cfg.writer.tbX_addImage('GT_' + data_info['img_name'], img_write)
            self.write_images -= 1

        img, label = self.lgtransformer.transpose(img, label)
        assert len(img) > 0, 'img length is error'
        assert len(label) > 0, 'lab length is error'

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
        _label = None
        while img is None or _label is None:  # if there is no data in img or label
            if self.one_test: index = 0
            data_info = self.dataset_infos[index]
            img, _label = self._read_datas(data_info)  # labels: (x1, y1, x2, y2) & must be absolutely labels.
            index = random.randint(0, len(self.dataset_infos) - 1)
            if not self.is_training and not _label:  # for test
                break

        label = np.asarray(_label[:], np.float32)
        # assert len(label.shape) == 2
        return img, label, data_info

    def _add_dataset(self, dataset, is_training):
        self.dataset_txt = dataset
        self.is_training = is_training

    def _load_labels2memery(self, dataset_txt, one_name, pre_load_labels):
        ftxt = 'train_' if self.is_training else 'test_'
        cachepath = os.path.join(self.cfg.PATH.INPUT_PATH, ftxt + ''.join(self.cfg.TRAIN.TRAIN_DATA_FROM_FILE[0]) + '.datacache')
        if os.path.isfile(cachepath) and not self.one_test:
            print('loading %s ...' % cachepath)
            data_infos = torch.load(cachepath)
            return data_infos
        if self.one_test:
            dataset_txt = one_name
        data_infos = []
        tqd = tqdm.tqdm(dataset_txt)

        def load_data_fun(data_line):
            x_path = os.path.join(self.cfg.PATH.INPUT_PATH, data_line[1])
            y_path = os.path.join(self.cfg.PATH.INPUT_PATH, data_line[2])
            this_data_info = {'img_name': data_line[0],
                              'img_path': x_path,
                              'lab_path': y_path,
                              'ratio(w,h)': [1, 1],
                              'padding(w,h)': [0, 0]
                              }
            if pre_load_labels:
                label_i, wh = self._load_labels(data_info=this_data_info)
            else:
                label_i = 'not_load'
                wh = None
            this_data_info['label'] = label_i
            this_data_info['wh_original'] = wh
            if label_i != []:
                data_infos.append(this_data_info)

        multiprocess = 1
        if multiprocess:
            all_task = []
            from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
            executor = ThreadPoolExecutor(max_workers=32)
            for data_line in tqd:
                all_task.append(executor.submit(load_data_fun, data_line))
            out = [future.result() for future in tqdm.tqdm(all_task)]
        else:
            for data_line in tqd:
                load_data_fun(data_line)

        if not self.one_test:
            print('saving %s ...' % cachepath)
            torch.save(data_infos, cachepath)
        return data_infos

    def _read_datas(self, data_info):
        '''
        Read images and labels depend on the idx.
        :param image: if there is a image ,the just return the image.
        :return: images, labels, image_size
        '''

        x_path = data_info['img_path']
        y_path = data_info['lab_path']
        if self.print_path:
            print(x_path, '<--->', y_path)

        if self.cfg.TRAIN.USE_LMDB:
            img_name = data_info['img_name']
            image_bin = self.txn_image.get(img_name.encode())
            image_buf = np.frombuffer(image_bin, dtype=np.uint8)
            img = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
        else:
            # if not (os.path.isfile(x_path) and os.path.isfile(y_path)):
            #     print('ERROR, NO SUCH A FILE.', x_path, '<--->', y_path)
            #     exit()
            img = cv2.imread(x_path, cv2.IMREAD_COLOR)

        # load labels
        label, wh = data_info['label'], data_info['wh_original']
        if label == 'not_load':
            label, wh = self._load_labels(data_info=data_info)
            data_info['label'], data_info['wh_original'] = label, wh
        if label == [] or label == [[]]:
            # print('loader obd :none label at:', data_info)
            label = None
        return img, label

    def _is_finedata(self, xyxy):
        '''
        area < min ?
        :param xyxy:
        :return:
        '''
        x1, y1, x2, y2 = xyxy
        for point in xyxy:
            if point < 0.: return False
        if x2 - x1 <= 0.: return False
        if y2 - y1 <= 0.: return False
        if (x2 - x1) * (y2 - y1) < self.cfg.TRAIN.MIN_AREA: return False
        return True

    def _load_lmdb_label_from_name(self, data_info):
        img_name = data_info['img_name']
        label_bin = self.txn_label.get(img_name.encode())
        label_infos = pickle.loads(label_bin)
        bboxes = label_infos['bboxes']
        cls_names = label_infos['cls_names']
        label = []
        for i, bbx in enumerate(bboxes):
            cls_name = cls_names[i]
            if self.class_name[cls_name] not in self.class_name:
                continue
            if not self._is_finedata(bbx): continue
            label.append([self.cls2idx[self.class_name[cls_name]], bbx[0], bbx[1], bbx[2], bbx[3]])
        return label

    def _load_labels(self, data_info=None, predicted_line=False, pass_obj=[]):
        """
        Parse the labels from file.

        :param pass_obj: pass the labels in the list.
                        e.g. pass_obj=['Others','Pedestrian', 'DontCare']
        :param path: the path of file that need to parse.
        :return:lists of the classes and the key points============ [x1, y1, x2, y2].
        """
        pass_obj = self.cfg.TRAIN.PASS_OBJ
        bbs = []
        path = data_info['lab_path']
        width = None
        height = None
        if self.cfg.TRAIN.USE_LMDB:
            bbs = self._load_lmdb_label_from_name(data_info)

        elif os.path.basename(path).split('.')[-1] == 'xml' and not predicted_line:
            utf8_parser = ET.XMLParser(encoding='utf-8')
            tree = ET.parse(path, parser=utf8_parser)
            root = tree.getroot()
            width = int(root.find('size').find('width').text)
            height = int(root.find('size').find('height').text)

            for obj in root.findall('object'):
                try:
                    difficult = int(obj.find('difficult').text) == 1
                except:
                    difficult = False
                if not self.keep_difficult and difficult:
                    continue

                cls_name = obj.find('name').text.strip().strip('\ufeff')
                if cls_name not in self.class_name:
                    if cls_name not in self.passedlabels:
                        self.passedlabels.append(cls_name)
                        print(cls_name, 'is passed')
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

        elif os.path.basename(path).split('.')[-1] == 'txt' and not predicted_line:
            file_open = open(path, 'r')
            for line in file_open.readlines():
                if 'UCAS_AOD' in self.cfg.TRAIN.TRAIN_DATA_FROM_FILE:
                    tmps = line.strip().split('\t')
                    box_x1 = float(tmps[9])
                    box_y1 = float(tmps[10])
                    box_x2 = box_x1 + float(tmps[11])
                    box_y2 = box_y1 + float(tmps[12])
                    if not self._is_finedata([box_x1, box_y1, box_x2, box_y2]): continue
                    bbs.append([1, box_x1, box_y1, box_x2, box_y2])

                elif 'VisDrone2019' in self.cfg.TRAIN.TRAIN_DATA_FROM_FILE:
                    name_dict = {'0': 'ignored regions', '1': 'pedestrian', '2': 'people',
                                 '3': 'bicycle', '4': 'car', '5': 'van', '6': 'truck',
                                 '7': 'tricycle', '8': 'awning-tricycle', '9': 'bus',
                                 '10': 'motor', '11': 'others'}
                    tmps = line.strip().split(',')
                    realname = name_dict[tmps[5]]
                    if realname not in self.class_name:
                        continue
                    if realname in pass_obj:
                        continue
                    box_x1 = float(tmps[0])
                    box_y1 = float(tmps[1])
                    box_x2 = box_x1 + float(tmps[2])
                    box_y2 = box_y1 + float(tmps[3])

                    if not self._is_finedata([box_x1, box_y1, box_x2, box_y2]): continue
                    bbs.append([self.cls2idx[self.class_name[realname]], box_x1, box_y1, box_x2, box_y2])
                elif 'TS02' in self.cfg.TRAIN.TRAIN_DATA_FROM_FILE:
                    tmps = line.strip().split(' ')
                    realname = 'car'
                    if realname not in self.class_name:
                        continue
                    if realname in pass_obj:
                        continue
                    img_w = 1920
                    img_h = 1080
                    x = float(tmps[1]) * img_w
                    y = float(tmps[2]) * img_h
                    w = float(tmps[3]) * img_w
                    h = float(tmps[4]) * img_h

                    box_x1 = x - w / 2.
                    box_y1 = y - h / 2.
                    box_x2 = x + w / 2.
                    box_y2 = y + h / 2.

                    if not self._is_finedata([box_x1, box_y1, box_x2, box_y2]): continue
                    bbs.append([self.cls2idx[self.class_name[realname]], box_x1, box_y1, box_x2, box_y2])
                else:
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

        elif 'COCO' in self.cfg.TRAIN.TRAIN_DATA_FROM_FILE:
            img_name = data_info[0].strip()
            img_info, ann_info = self.coco.prepare_data(img_name)
            for i, bbx in enumerate(ann_info['bboxes']):
                cls_name = ann_info['labels'][i]
                if cls_name not in self.class_name:
                    print(cls_name, 'is passed')
                    continue
                if self.class_name[cls_name] in pass_obj:
                    continue
                if not self._is_finedata(bbx): continue
                bbs.append([self.cls2idx[self.class_name[cls_name]], bbx[0], bbx[1], bbx[2], bbx[3]])

        return bbs, [width, height]

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

        labels = self.collect_fun_normal(labels, infos)

        return [imgs, labels, list(infos)]

    def collect_fun_normal(self, labels, infos):
        for i, label in enumerate(labels):
            try:
                label[:, 0] = i
            except:
                print('collate fun error！！！：', infos[i])
        try:
            labels = torch.cat(labels, 0)
        except:
            print('collate fun error！！！：', labels, infos)
        return labels

    def collect_fun_fcos(self, labels, infos):  # not use
        pad_boxes_list = []
        pad_classes_list = []
        max_num = 0
        for i, label in enumerate(labels):
            n = label.shape[0]
            if n > max_num: max_num = n

        for i, label in enumerate(labels):
            box = label[..., 2:]
            cls = label[..., 1]
            pad_boxes_list.append(torch.nn.functional.pad(box, (0, 0, 0, max_num - label.shape[0]), value=-1))
            pad_classes_list.append(torch.nn.functional.pad(cls, (0, max_num - cls.shape[0]), value=-1))

        batch_boxes = torch.stack(pad_boxes_list)
        batch_classes = torch.stack(pad_classes_list)

        return batch_boxes, batch_classes
