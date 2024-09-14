import os
import cv2
import glob
from lgdet.util.util_show_img import _show_img


def show_tracker_OTB(img_path, gt_path):
    img_list = glob.glob(img_path+'*.jpg')
    lines = open(gt_path).readlines()
    for img, line in zip(img_list, lines):
        img = cv2.imread(img)
        box = line.strip().split(',')
        box[0] = int(box[0])
        box[1] = int(box[1])
        box[2] = box[0]+int(box[2])
        box[3] = box[1]+int(box[3])

        boxes=['OTB_GT']
        boxes.append(box)
        _show_img(img, [boxes],show_time=200)



if __name__ == '__main__':
    dataset = 'E:/datasets/TRACK/OTB100/'
    obj_name = 'BlurCar1'
    img_path = os.path.join(dataset, obj_name + '/img/')
    gt_path = os.path.join(dataset, obj_name +'/groundtruth_rect.txt')

    show_tracker_OTB(img_path, gt_path)