from torch.utils.data import DataLoader
import numpy as np
import os
import torch
import cv2
from glob import glob


class MyDataset(DataLoader):
    """
    Load data with DataLoader.
    """

    def __init__(self, img_folder, lab_folder, transform=None):
        self.transform = transform
        ll = glob(img_folder)
        # read labels first, if there is no labels.
        lab_names = os.listdir(lab_folder)
        self.imgs = []
        for id, lab_name in enumerate(lab_names):
            file_first_name = str(lab_name).split('.')[0]
            lab_abspath = os.path.join(lab_folder, lab_name)
            bbs = self._read_line(lab_abspath)
            if bbs is []:
                continue
            img_abspath = os.path.join(img_folder, file_first_name + '.png')
            self.imgs.append((img_abspath, bbs))

    def __getitem__(self, index):
        fn, label = self.imgs[index]

        img = cv2.imread(fn)
        img = cv2.resize(img, (400, 400))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

    def _read_line(self, path, pass_obj=['Others', ]):
        """
        Parse the labels from file.

        :param pass_obj: pass the labels in the list.
                        e.g. pass_obj=['Others','Pedestrian']
        :param path: the path of file that need to parse.
        :return:lists of the classes and the key points.
        """
        file_open = open(path, 'r')
        bbs = []
        for line in file_open.readlines():
            tmps = line.strip().split(' ')
            if tmps[0] in pass_obj:
                continue
            box_x1 = float(tmps[4])
            box_y1 = float(tmps[5])
            box_x2 = float(tmps[6])
            box_y2 = float(tmps[7])
            bbs.append([tmps[0], box_x1, box_y1, box_x2, box_y2])
        return bbs


def collate_fn_1(batches):
    imgs = []
    labs = []
    for i, batch in enumerate(batches):
        m = batch[0]
        imgs.append(m)
        labs.append(batch[1])
    images = torch.from_numpy(np.asarray(imgs))
    return images, labs


def collate_fn_2(batch):
    imgs, labels = zip(*batch)
    return torch.from_numpy(np.asarray(imgs)), list(labels)


IMG_PATH = 'E://datasets//kitti//training//images//'
LAB_PATH = 'E://datasets//kitti//training//labels//'
mydata = MyDataset(IMG_PATH,LAB_PATH)

traindata = DataLoader(dataset=mydata, batch_size=10, collate_fn=collate_fn_1, shuffle=False)

data = iter(traindata)
i, l = next(data)  # 其中default_collate会将labels分割合并转换成tensor。collate_fn：如何取样本的，我们可以定义自己的函数来准确地实现想要的功能
# or use the next code.
for i, data in enumerate(traindata):
    print(data[0].size, data[1].shape)
