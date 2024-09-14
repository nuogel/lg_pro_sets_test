'''
the performance on the test dataset is very good. will be useful !
'''
import cv2
from kcf import Tracker
from argparse import ArgumentParser
import os
import numpy as np
import glob


def load_bbox(ground_file, resize, dataformat=0):
    f = open(ground_file)
    lines = f.readlines()
    bbox = []
    for line in lines:
        if line:
            pt = line.strip().split(',')
            pt_int = [float(ii) for ii in pt]
            bbox.append(pt_int)
    bbox = np.array(bbox)
    if resize:
        bbox = (bbox.astype('float32') / 2).astype('int')
    else:
        bbox = bbox.astype('float32').astype('int')
    if dataformat:
        bbox[:, 2] = bbox[:, 0] + bbox[:, 2]
        bbox[:, 3] = bbox[:, 1] + bbox[:, 3]
    return bbox


def load_imglst(img_dir):
    file_lst = [pic for pic in os.listdir(img_dir) if '.jpg' in pic]
    img_lst = [os.path.join(img_dir, filename) for filename in file_lst]
    return img_lst


if __name__ == '__main__':
    OTB100 = 1
    LG = 0

    if OTB100:
        dataset = 'E:/datasets/TRACK/OTB100/BlurCar2/'
        save_directory = './'
        show_result = 1
        img_lst = load_imglst(dataset + '/img/')
        bbox_lst = load_bbox(os.path.join(dataset + '/groundtruth_rect.txt'), 0, dataformat=0)
        x1, y1, w, h = bbox_lst[0]
        roi = (x1, y1, w, h)
    elif LG:
        img_path = 'E:\datasets\TRACK\LG\images/img_'
        img_lst = []
        for i in range(70, 967):
            img_lst.append(img_path + str(i) + '.jpg')
    frames = len(img_lst)
    tracker = Tracker()
    frame1 = cv2.imread(img_lst[0])
    roi = cv2.selectROI("tracking", frame1, False, False)
    tracker.init(frame1, roi)
    j = 1
    for i in range(1, frames):
        frame = cv2.imread(img_lst[i])
        if frame is None:
            continue
        (x1, y1, w, h), max_response = tracker.update(frame, scales=[0.95, 1.0, 1.05])
        min_response = 0.009
        print('max_response', max_response)
        repare_step = 20000
        if max_response < min_response or j % repare_step == 0:

            if LG:
                print('boxing a box.')
                roi = cv2.selectROI("tracking", frame, False, False)
            if OTB100:
                roi = bbox_lst[i]
            if max_response < min_response: print('max_response：', max_response, 'repare bbox', (x1, y1, w, h), '->', roi)
            if j % 10 == 0: print('check bbox', j)
            x1 = (x1 + roi[0]) // 2
            y1 = (y1 + roi[1]) // 2
            w = (w + roi[2]) // 2
            h = (h + roi[3]) // 2
            tracker.init(frame, (x1, y1, w, h))
            j = 1

        j += 1
        cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 255), 1)
        cv2.imshow('tracking', frame)
        c = cv2.waitKey(1) & 0xFF
        if c == 27 or c == ord('q'):
            break
    cv2.destroyAllWindows()
