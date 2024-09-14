from torch.utils.data import DataLoader
import numpy as np
import os
import torch
import cv2
from cfg import config as cfg
from lgdet.util.util_read_label_xml import _read_label_voc


class MyDataset_VOC(DataLoader):
    """
    Load data with DataLoader.
    """

    def __init__(self, cfg, split='train', crop=None, flip=False, transform=None):
        self.crop = crop
        self.flip = flip
        self.transform = transform
        self.inputs = []
        self.targets = []

        voc_path = cfg.VOC_PATH
        year = cfg.YEAR
        assert split in ['train', 'val'], 'ERROR: no such a split.'
        name_txt_path = os.path.join(voc_path, 'VOC%d' % year, 'ImageSets//Main', split + '.txt')
        imgs_path = os.path.join(voc_path, 'VOC%d' % year, 'JPEGImages')
        labs_path = os.path.join(voc_path, 'VOC%d' % year, 'Annotations')

        f = open(name_txt_path, 'r')
        lines = f.readlines()
        for line in lines:
            name = line.split('\n')[0]
            imgs_path_list = os.path.join(imgs_path, name + '.jpg')
            labs_path_list = os.path.join(labs_path, name + '.xml')
            self.inputs.append(imgs_path_list)
            self.targets.append(labs_path_list)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        img_path, lab_path = self.inputs[index], self.targets[index]
        img = cv2.imread(img_path)
        img = cv2.resize(img, (400, 400))
        lab = _read_label_voc(lab_path)[0]
        if self.transform is not None:
            img, lab = self.transform(img, lab)
        return img, lab  # only need the labels


#
def collate_fn(batch):
    '''
    collate_fn：如何取样本的，我们可以定义自己的函数来准确地实现想要的功能
    其中default_collate会将labels分割合并转换成tensor。
    !!!***if not use my own collect_fun ,the labels will be wrong orders.***
    :param batch:
    :return:
    '''
    imgs, labels = zip(*batch)
    return torch.from_numpy(np.asarray(imgs)), list(labels)


batch_size = 10
mydata = MyDataset_VOC(cfg)
traindata = DataLoader(dataset=mydata, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

data = iter(traindata)
i, l = next(data)
# or use the next code.
for i, (img, lab) in enumerate(traindata):
    print(lab)


#######################################################################################################
###########               e.g. 2               #######################################################
######################################################################################################

