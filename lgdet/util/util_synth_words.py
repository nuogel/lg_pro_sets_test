import os
import random
import tqdm
import numpy as np
from imgaug import augmenters as iaa

import cv2
imgdir = '/media/dell/data/ocr/计费清单识别/集团-计费清单/images'
labdir = '/media/dell/data/ocr/计费清单识别/集团-计费清单/labels'
saveimgdir = '/media/dell/data/ocr/计费清单识别/synthdata_from_raw/images'
savelabdir = '/media/dell/data/ocr/计费清单识别/synthdata_from_raw/labels'

def data_aug(img):
    base_funs = [iaa.Grayscale(alpha=(0, 1)),
                 iaa.Add((-50, 50)),
                 iaa.Dropout(0.08, per_channel=0.5),

                 iaa.MotionBlur(k=(3, 11), angle=[-45, 45]),
                 # 高斯模糊
                 iaa.GaussianBlur((0, 3)),
                 ]
    aug_funs = random.choice(base_funs)
    seq_det = iaa.Sequential(aug_funs)
    seq_det = seq_det.to_deterministic()
    img = seq_det.augment_images(img)
    return img



def makedata():
    for imgp in tqdm.tqdm(os.listdir(imgdir)):
        labp = os.path.join(labdir, imgp.replace('.jpg', '.txt'))
        savelabp = os.path.join(savelabdir, imgp.replace('.jpg', '.txt'))
        saveimgp = os.path.join(saveimgdir, imgp)

        imgp = os.path.join(imgdir, imgp)
        img = cv2.imread(imgp)
        h, w, c = img.shape
        labf = open(labp, 'r')
        boxtxt = labf.read()
        labf.close()

        writelabf = open(savelabp, 'w')
        for i in range(50):
            fontlist = [0, 2, 3, 4, 6, 7]
            x = random.randint(100, w - 100)
            y = random.randint(100, h - 100)
            # x,y  = 300,300
            point = (x, y)
            font_i = random.choice(fontlist)
            scale = 2
            colorint = random.randint(40, 80)
            word = str(random.randint(0, 9))
            img = cv2.putText(img, word, (x, y), font_i, fontScale=1.2, color=(colorint, colorint, colorint), thickness=2)
            hight = 32
            width = 25
            box = [x - 1, y - hight, x + width, y + 5]
            # img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=(0, 255, 0))
            imgcorp = img[box[1]:box[3], box[0]:box[2], :]
            # imgcorp = data_aug(imgcorp)
            boxtxt += ','.join(map(str, [box[0], box[1], box[2], box[1], box[2], box[3], box[0], box[3]])) + ',' + word + '\n'

            # -hao

            x = random.randint(100, w - 200)
            y = random.randint(100, h - 200)
            # x,y  = 300,300
            font_i = random.choice(fontlist)
            scale = 2
            colorint = random.randint(40, 80)
            word = '-'+ str(random.random())[:random.randint(3,7)]
            lenword = len(word)
            img = cv2.putText(img, word, (x, y), font_i, fontScale=1.2, color=(colorint, colorint, colorint), thickness=2)
            hight = 32
            width = 24*lenword
            box = [x - 1, y - hight, x + width, y + 5]
            # img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=(0, 255, 0))
            imgcorp = img[box[1]:box[3], box[0]:box[2], :]
            # imgcorp = data_aug(imgcorp)
            boxtxt += ','.join(map(str, [box[0], box[1], box[2], box[1], box[2], box[3], box[0], box[3]])) + ',' + word + '\n'

        cv2.imwrite(saveimgp, img )
        writelabf.write(boxtxt)
        writelabf.close()
        # r =2048/ max(img.shape)
        # img = cv2.resize(img, None, fx=r, fy=r)
        # cv2.imshow('img', img)
        # cv2.waitKey()


def show_data():
    for imgp in os.listdir(saveimgdir):
        imgpi = os.path.join(saveimgdir, imgp)
        img = cv2.imread(imgpi)
        labpi = os.path.join(savelabdir,imgp.replace('.jpg', '.txt'))
        f = open(labpi)
        for line in f.readlines():
            tmp  = line.split(',')
            box = tmp[:-1]
            lab = tmp[-1].strip()
            boxarray = np.asarray(list(map(int, box))).reshape(-1, 2)
            cv2.polylines(img, [boxarray], isClosed=True,color=(255,0,0), thickness=2)
            img = cv2.putText(img, lab, (boxarray[0][0], boxarray[0][1]), 1, fontScale=1.2, color=(255,0,0), thickness=2)

        r =2048/ max(img.shape)
        img = cv2.resize(img, None, fx=r, fy=r)
        cv2.imshow('img', img)
        cv2.waitKey()


# makedata()
show_data()
