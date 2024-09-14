import os
import glob
import cv2
import numpy as np

paths = glob.glob('/media/dell/data/比赛/国家电网识图比赛/训练集/tmp/ocr/train/*.png')

size = []
for path in paths:
    img = cv2.imread(path)
    size.append([min(img.shape[:2]), max(img.shape[:2])])

print(size, '\n', np.asarray(size)[0].mean(),np.asarray(size)[1].mean())  # =28.18190476190476
