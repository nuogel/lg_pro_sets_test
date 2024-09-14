import os
import random
from PIL import Image
import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2

CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
index_map = dict(zip(CLASSES, range(len(CLASSES))))


def voc_colormap(N=21):  # generate a color map of voc segmentation
    def bitget(val, idx):
        return ((val & (1 << idx)) != 0)

    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= (bitget(c, 0) << 7 - j)
            g |= (bitget(c, 1) << 7 - j)
            b |= (bitget(c, 2) << 7 - j)
            c >>= 3
        print(i, ':', [r, g, b])
        cmap[i, :] = [r, g, b]
    return cmap


VOC_COLORMAP = voc_colormap()


class VOC2012Dataset(DataLoader):
    def __init__(self, split='train', crop=None, flip=False):
        # super().__init__()
        self.crop = crop
        self.flip = flip
        self.inputs = []
        self.targets = []

        seg_train_path = 'E:\datasets\VOCdevkit\VOC2012\ImageSets\Segmentation//%s.txt' % split
        seg_img_path = 'E:\datasets\VOCdevkit\VOC2012\JPEGImages'
        seg_lab_path = 'E:\datasets\VOCdevkit\VOC2012\SegmentationClass'

        f = open(seg_train_path, 'r')
        lines = f.readlines()
        for line in lines:
            name = line.split('\n')[0]
            img_raw_path = os.path.join(seg_img_path, name + '.jpg')
            img_seg_path = os.path.join(seg_lab_path, name + '.png')
            self.inputs.append(img_raw_path)
            self.targets.append(img_seg_path)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        cls_num = 21

        def make_one_hot(i, N):
            one_hot = np.zeros(N)
            one_hot[i] = 1
            return one_hot

        input_cv = cv2.imread(self.inputs[i])
        target_cv = cv2.imread(self.targets[i])
        # cv2.imshow('img', target_cv)
        # cv2.waitKey()

        input_cv = np.asarray(cv2.cvtColor(input_cv, cv2.COLOR_BGR2RGB))
        target_cv = np.asarray(cv2.cvtColor(target_cv, cv2.COLOR_BGR2RGB))
        print(input_cv[0, 0], target_cv[150, 200])
        if self.crop:
            input_cv = input_cv[:self.crop, :self.crop]
            target_cv = target_cv[:self.crop, :self.crop]

        target_onehot = np.zeros((target_cv.shape[0], target_cv.shape[1], cls_num))
        for i, col in enumerate(VOC_COLORMAP):
            mask = (target_cv == col).sum(-1) == 3
            target_onehot[mask] = make_one_hot(i, cls_num)  # make the color to one hot matrix.

        return input_cv, target_onehot  # Return x, y (one-hot), y (index)


def collect_fun(batches):
    imgs = []
    labs = []
    for i, batch in enumerate(batches):
        imgs.append(batch[0])
        labs.append(batch[1])
    images = np.array(imgs)
    labels = np.array(labs)

    return images, labels


def decode_onehot_color(seg_onehot):
    if len(seg_onehot.shape) == 3:
        seg_onehot = [seg_onehot]
    assert len(seg_onehot.shape) == 4, 'shape error'

    for i, seg_img in enumerate(seg_onehot):
        seg_color = np.zeros((seg_img.shape[0], seg_img.shape[1], 3))
        seg_max = np.argmax(seg_img, -1)
        for i, color in enumerate(VOC_COLORMAP):
            mask = seg_max == i
            seg_color[mask] = color
        seg_color = np.array(seg_color, dtype=np.uint8)
        seg_color = cv2.cvtColor(seg_color, cv2.COLOR_RGB2BGR)
        cv2.imshow('img', seg_color)
        cv2.waitKey()


if __name__ == "__main__":
    batch_size = 1
    mydata = VOC2012Dataset()
    traindata = DataLoader(dataset=mydata, batch_size=batch_size, collate_fn=collect_fun, shuffle=False)

    # data = iter(traindata)
    # i, l,_ = next(data)
    # or use the next code.
    for i, (input, target) in enumerate(traindata):
        decode_onehot_color(target)
