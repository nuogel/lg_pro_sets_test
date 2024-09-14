import cv2
import glob
import os
from others.track.deep_sort.deep_sort import DeepSort
import numpy as np
import matplotlib.pyplot as plt
import random
from lgdet.util.util_make_VisDrone2019_VID_dataset import make_VisDrone2019_VID_dataset


def run_deep_sort_imglabels():
    img_path = '../KCF/example3/demo_result/images'
    lab_path = '../KCF/example3/demo_result/labels'
    for img_p in range(1, len(os.listdir(img_path))):
        id = '%05d' % img_p
        # id = os.path.basename(img_p).split('.')[0]
        imgpath_i = os.path.join(img_path, id + '.jpg')
        frame = cv2.imread(imgpath_i)
        lab_p = os.path.join(lab_path, id + '.txt')
        lab_id = []
        conf = []
        cls = []
        for line in open(lab_p).readlines():
            tmps = line.strip().split(',')
            l = float(tmps[0])
            t = float(tmps[1])
            w = float(tmps[2])
            h = float(tmps[3])

            x = l + w / 2
            y = t + h / 2
            # w = (x2 - x1)
            # h = (y2 - y1)

            lab_id.append([x, y, w, h])
            conf.append(1)
            cls.append(1)
        if lab_id == []: continue
        lab_id = np.asarray(lab_id)

        tracked_detections = deep_sort.update(lab_id, conf, frame)
        if tracked_detections == []: continue
        _show_track(tracked_detections, frame)


def run_deep_sort_AUTOAIR():
    img_path = 'F:\Projects\\auto_Airplane\TS02\\20191217_153659'
    lab_path = 'F:\Projects\\auto_Airplane\TS02\\20191217_153659_predicted_labels'
    for img_p in glob.glob(img_path + '/*.*'):
        id = os.path.basename(img_p).split('.')[0]
        frame = cv2.imread(img_p)
        lab_p = os.path.join(lab_path, id + '.txt')
        lab_id = []
        conf = []
        cls = []
        for line in open(lab_p).readlines():
            tmps = line.strip().split(' ')
            score = float(tmps[1])
            x1 = float(tmps[2])
            y1 = float(tmps[3])
            x2 = float(tmps[4])
            y2 = float(tmps[5])

            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            w = (x2 - x1)
            h = (y2 - y1)

            lab_id.append([x, y, w, h])
            conf.append(score)
            cls.append(1)
        if lab_id == []: continue
        lab_id = np.asarray(lab_id)

        tracked_detections = deep_sort.update(lab_id, conf, frame)
        if tracked_detections == []: continue
        _show_track(tracked_detections, frame)


def run_deep_sort_VISDRONE():
    all_info_list = make_VisDrone2019_VID_dataset(path='E:/datasets/VisDrone2019/VisDrone2019-VID-train/', show_images=False)
    for info_dict in all_info_list:
        for k, v in info_dict.items():
            img_path = v[0]
            images = cv2.imread(img_path)
            lab = np.asarray(v[1:])
            mask = lab[..., 0] == 2
            _lab = lab[mask]  # get people only
            _lab = _lab[..., 1:]
            _lab_xywh = np.zeros(_lab.shape)
            _lab_xywh[:, 0] = (_lab[:, 0] + _lab[:, 2]) / 2
            _lab_xywh[:, 1] = (_lab[:, 1] + _lab[:, 3]) / 2
            _lab_xywh[:, 2] = _lab[:, 2] - _lab[:, 0]
            _lab_xywh[:, 3] = _lab[:, 3] - _lab[:, 1]

            conf = np.ones(len(_lab_xywh))
            tracked_detections = deep_sort.update(_lab_xywh, conf, images)
            if tracked_detections == []: continue

            _show_track(tracked_detections, images)


def _show_track(tracked_detections, frame):
    print(tracked_detections)

    class_dict = {1: 'car', 2: 'person'}
    for x1, y1, x2, y2, obj_id in tracked_detections:
        color = bbox_palette[int(obj_id) % len(bbox_palette)]
        color = [int(i * 255) for i in color]
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        cv2.putText(frame, '{}-{}'.format(class_dict[1], int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    cv2.imshow('img', frame)
    cv2.waitKey()


if __name__ == '__main__':
    cmap = plt.get_cmap('tab20b')
    bbox_palette = [cmap(i)[:3] for i in np.linspace(0, 1, 100)]
    random.shuffle(bbox_palette)

    model_path = '/home/dell/lg/code/lg_pro_sets/saved/checkpoint/ckpt.t7'
    deep_sort = DeepSort(model_path=model_path, )
    run_deep_sort_imglabels()
