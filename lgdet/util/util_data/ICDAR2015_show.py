import os

import cv2
import numpy as np

base_dir = '/media/dell/data/ocr/ICDAR2015'
img_base_path = 'ch4_test_images' #'ch4_training_images'
lab_base_path='Challenge4_Test_Task1_GT'#'ch4_training_localization_transcription_gt'


img_dir = os.path.join(base_dir,img_base_path)
lab_dir = os.path.join(base_dir,lab_base_path)

for imgi in os.listdir(img_dir):
    img_path = os.path.join(img_dir, imgi)
    lab_path = os.path.join(lab_dir, 'gt_'+imgi.split('.')[0]+'.txt')
    img = cv2.imread(img_path)
    lines = open(lab_path, 'r', encoding='utf-8-sig').readlines()
    boxes = []
    for line in lines:
        tmp = line.strip().split(',')
        if tmp[-1]=='':tmp = tmp[:-1]
        box = np.asarray([int(x) for x in tmp[:-1]]).reshape([-1,2])
        boxes.append(box)
        name = tmp[-1]
        img = cv2.putText(img, name, (box[0, 0], box[0, 1]), fontFace=1, fontScale=0.8, color=(0,255,0))
        img = cv2.polylines(img, [box], isClosed=True, color=(255,0,0))
    cv2.imshow(imgi.split('.')[0],img)
    cv2.waitKey()
    cv2.destroyAllWindows()

