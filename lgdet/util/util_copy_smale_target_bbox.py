# copy smale target with bounding boxes to background pics.
import os
from util_load_xml2bbox import GetXmlGtBoxes
import cv2
import random
import numpy as np
from util_iou import box_iou
import tqdm

from lxml import etree


class GEN_Annotations:
    def __init__(self, filename='voc xml'):
        self.root = etree.Element("annotation")
        child1 = etree.SubElement(self.root, "folder")
        child1.text = "VOC2007"
        child2 = etree.SubElement(self.root, "filename")
        child2.text = filename
        child3 = etree.SubElement(self.root, "source")
        child4 = etree.SubElement(child3, "annotation")
        child4.text = "PASCAL VOC2007"
        child5 = etree.SubElement(child3, "database")
        child5.text = "Unknown"
        child6 = etree.SubElement(child3, "image")
        child6.text = "flickr"
        # child7 = etree.SubElement(child3, "flickrid")
        # child7.text = "35435"

    def set_size(self, witdh, height, channel):
        size = etree.SubElement(self.root, "size")
        widthn = etree.SubElement(size, "width")
        widthn.text = str(witdh)
        heightn = etree.SubElement(size, "height")
        heightn.text = str(height)
        channeln = etree.SubElement(size, "depth")
        channeln.text = str(channel)

    def savefile(self, filename):
        tree = etree.ElementTree(self.root)
        tree.write(filename, pretty_print=True, xml_declaration=False, encoding='utf-8')

    def add_pic_attr(self, label, xmin, ymin, xmax, ymax):
        object = etree.SubElement(self.root, "object")
        namen = etree.SubElement(object, "name")
        namen.text = label
        pose = etree.SubElement(object, "pose")
        pose.text = str(0)
        truncated = etree.SubElement(object, "truncated")
        truncated.text = str(0)
        difficult = etree.SubElement(object, "difficult")
        difficult.text = str(0)
        bndbox = etree.SubElement(object, "bndbox")
        xminn = etree.SubElement(bndbox, "xmin")
        xminn.text = str(xmin)
        yminn = etree.SubElement(bndbox, "ymin")
        yminn.text = str(ymin)
        xmaxn = etree.SubElement(bndbox, "xmax")
        xmaxn.text = str(xmax)
        ymaxn = etree.SubElement(bndbox, "ymax")
        ymaxn.text = str(ymax)

    def genvoc(self, filename, class_, width, height, depth, xmin, ymin, xmax, ymax, savedir):
        anno = GEN_Annotations(filename)
        anno.set_size(width, height, depth)
        anno.add_pic_attr("pos", xmin, ymin, xmax, ymax)
        anno.savefile(savedir)

def box_iou(boxes1, boxes2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(boxes1.t())
    area2 = box_area(boxes2.t())

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    inter = (rb - lt).clamp(min=0).prod(2)  # [N,M]
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

class CopyLittleTarget:
    def __init__(self, imgs_path, labs_path, save_targe_folder, name_dict1, name_dict2, copy_number, save_xml_path):
        self.imgs_path = imgs_path
        self.labs_path = labs_path
        self.name_dict1 = name_dict1
        self.name_dict2 = name_dict2
        self.copy_number = copy_number
        self.parse_xml = GetXmlGtBoxes(keep_difficult=False, name_dict=self.name_dict1)
        self.img_list = os.listdir(self.imgs_path)
        self.save_xml_path = save_xml_path
        self.save_targe_folder = save_targe_folder
        self.remake_targets = False

    def run(self):
        if self.remake_targets:
            self._make_source_targets()

        targets_dict = {}
        targets_dict["烟雾"] = [os.path.join(self.save_targe_folder, "烟雾", t) for t in
                              os.listdir(os.path.join(self.save_targe_folder, name_dict2[1]))]
        targets_dict["火"] = [os.path.join(self.save_targe_folder, "火", t) for t in
                             os.listdir(os.path.join(self.save_targe_folder, name_dict2[2]))]

        for ii, img_name in enumerate(self.img_list):

            dest_img, dest_bboxes = self._make_dest_images(img_name, targets_dict)
            print('deeling with pic:', ii, '-', img_name)
            dest_lab_path = os.path.join(self.save_xml_path, 'labels', img_name.replace('.jpg', '.xml'))
            dest_img_path = os.path.join(self.save_xml_path, 'images', img_name)
            cv2.imwrite(dest_img_path, dest_img)
            self.gen_xml = GEN_Annotations(dest_lab_path)
            size = dest_img.shape
            self.gen_xml.set_size(size[1], size[0], size[2])
            for bbox in dest_bboxes:
                self.gen_xml.add_pic_attr(self.name_dict2[bbox[0]], bbox[1], bbox[2], bbox[3], bbox[4])
                self.gen_xml.savefile(dest_lab_path)

    def _make_source_targets(self):
        print('collecting little targets ...')
        for source_img_name in tqdm.tqdm(self.img_list):
            source_img_path = os.path.join(self.imgs_path, source_img_name)
            source_lab_path = os.path.join(self.labs_path, source_img_name.replace('.png', '.xml'))
            source_img = cv2.imread(source_img_path)
            source_bboxes = self.parse_xml.get_groundtruth(source_lab_path)
            if source_bboxes == []: continue
            des_box = np.asarray(source_bboxes)[:, 1:5]
            iou = box_iou(des_box, des_box)
            mask = np.sum(iou > 0, -1) == 1
            ok_box = np.asarray(source_bboxes)[mask]
            for i, box in enumerate(ok_box):
                x1, y1, x2, y2 = box[1:5]
                area = (x2 - x1) * (y2 - y1)
                if area < 10:
                    continue

                source_t_i = source_img[y1:y2, x1:x2, :]
                save_name = os.path.join(self.save_targe_folder, self.name_dict2[box[0]],
                                         source_img_name.split('.')[0] + '_%03d.jpg' % i)
                cv2.imwrite(save_name, source_t_i)

    def _crop_source_targets(self):
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
                save_name = os.path.join(self.save_targe_folder, name,
                                         source_img_name.split('.')[0] + '_%03d.jpg' % i)
                os.makedirs(os.path.dirname(save_name), exist_ok=True)
                cv2.imwrite(save_name, source_t_i)
        print(len(name_list), name_list)
        list(name_list).sort()
        for it in name_list:
            print(it)

    def _make_dest_images(self, dest_img, targets_dict):
        dest_img_path = os.path.join(self.imgs_path, dest_img)
        dest_lab_path = os.path.join(self.labs_path, dest_img.replace('.jpg', '.xml'))
        dest_img = cv2.imread(dest_img_path)
        dest_bboxes = self.parse_xml.get_groundtruth(dest_lab_path)
        d_size = dest_img.shape
        for k, v in targets_dict.items():
            if len(v) >= self.copy_number:
                s_targets = random.sample(v, self.copy_number)
            else:
                s_targets = v

            for s_target_p in s_targets:
                s_target = cv2.imread(s_target_p)
                s_size = s_target.shape
                x1 = random.randint(0, d_size[1] - s_size[1])
                y1 = random.randint(0, d_size[0] - s_size[0])
                iou_max = 0
                if len(dest_bboxes) > 0:
                    s_box = np.asarray([[x1, y1, x1 + s_size[1], y1 + s_size[0]]])
                    des_box = np.asarray(dest_bboxes)[:, 1:5]
                    iou = box_iou(s_box, des_box)
                    iou_max = iou.max()

                if iou_max == 0:
                    dest_img[y1:y1 + s_size[0], x1:x1 + s_size[1]] = s_target
                    dest_bboxes.append([self.name_dict1[k], x1, y1, x1 + s_size[1], y1 + s_size[0]])

        return dest_img, dest_bboxes


if __name__ == '__main__':
    imgs_path = '/media/dell/data/比赛/第一批标注数据35张/images'
    labs_path = '/media/dell/data/比赛/第一批标注数据35张/labels'
    save_xml_path = '/media/dell/data/比赛/第一批标注数据35张/crop'
    save_targe_folder = os.path.join(save_xml_path, 'smale_targets')

    name_dict1 = {"烟雾": 1, '﻿火': 2, "火": 2, 'person': 3}
    name_dict2 = {1: "烟雾", 2: "火"}
    copy_number = 20

    clt = CopyLittleTarget(imgs_path, labs_path, save_targe_folder, name_dict1, name_dict2, copy_number, save_xml_path)
    # clt.run()

    clt._crop_source_targets()
