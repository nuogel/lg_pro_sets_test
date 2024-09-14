import cv2
import glob
import os
from others.track.sort.sort import Sort
import numpy as np
import matplotlib.pyplot as plt
import random


def test():
    cmap = plt.get_cmap('tab20b')
    bbox_palette = [cmap(i)[:3] for i in np.linspace(0, 1, 1000)]
    random.shuffle(bbox_palette)
    class_dict = {1: 'car'}
    img_path = 'F:\Projects\\auto_Airplane\TS02\\20191217_153659'
    lab_path = 'F:\Projects\\auto_Airplane\TS02\\20191217_153659_predicted_labels'
    sort = Sort()
    for img_p in glob.glob(img_path + '/*.*'):
        id = os.path.basename(img_p).split('.')[0]
        frame = cv2.imread(img_p)
        lab_p = os.path.join(lab_path, id + '.txt')
        lab_id = []
        for line in open(lab_p).readlines():
            tmps = line.strip().split(' ')
            score = float(tmps[1])
            x1 = float(tmps[2])
            y1 = float(tmps[3])
            x2 = float(tmps[4])
            y2 = float(tmps[5])
            lab_id.append([x1, y1, x2, y2, score, 1, 1])
        if lab_id == []: continue
        lab_id = np.asarray(lab_id)
        tracked_detections = sort.update(lab_id)
        print(tracked_detections)

        for x1, y1, x2, y2, obj_id, cls_pred in tracked_detections:
            label = class_dict[int(cls_pred)]
            color = bbox_palette[int(obj_id) % len(bbox_palette)]
            color = [int(i * 255) for i in color]
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            # cv2.rectangle(frame, (x1, y1 - 10), (x1 + len(label) * 19 + 30, y1), color, -1)
            cv2.putText(frame, '{}-{}'.format(label, int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.imshow('img', frame)
        cv2.waitKey()


test()
