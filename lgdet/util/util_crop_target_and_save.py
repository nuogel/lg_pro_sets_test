# copy smale target with bounding boxes to background pics.
import os
from util_load_xml2bbox import GetXmlGtBoxes
import cv2
import random
import numpy as np
from util_iou import box_iou
import tqdm

from lxml import etree


class CopyLittleTarget:
    def __init__(self, imgs_path, labs_path, save_targe_folder):
        self.imgs_path = imgs_path
        self.labs_path = labs_path
        self.parse_xml = GetXmlGtBoxes()
        self.img_list = os.listdir(imgs_path)
        self.save_targe_folder = save_targe_folder

    def crop_source_targets(self):
        print('collecting little targets ...')
        name_list = set()
        for source_img_name in tqdm.tqdm(self.img_list):
            source_img_path = os.path.join(self.imgs_path, source_img_name)
            source_lab_path = os.path.join(self.labs_path, source_img_name.replace('.jpg', '.xml'))
            source_img = cv2.imread(source_img_path)
            source_bboxes = self.parse_xml.get_groundtruth(source_lab_path)
            if source_bboxes == []: continue
            for i, box in enumerate(source_bboxes):
                x1, y1, x2, y2 = box[1:5]
                name = box[0]
                name_list.add(name)
                source_t_i = source_img[y1:y2, x1:x2, :]
                save_name = os.path.join(self.save_targe_folder, name, source_img_name.split('.')[0] + '_%03d.jpg' % i)
                os.makedirs(os.path.dirname(save_name), exist_ok=True)
                cv2.imwrite(save_name, source_t_i)
        print(len(name_list), name_list)
        list(name_list).sort()
        for it in name_list:
            print(it)




if __name__ == '__main__':
    imgs_path = '/media/dell/data/sleep/长寿-睡岗-3.1-3.9/images'
    labs_path = '/media/dell/data/sleep/长寿-睡岗-3.1-3.9/labels'
    save_targe_folder = '/media/dell/data/sleep/睡岗-Corp'

    clt = CopyLittleTarget(imgs_path, labs_path, save_targe_folder)
    clt.crop_source_targets()
