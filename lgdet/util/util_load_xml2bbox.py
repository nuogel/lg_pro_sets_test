import random
import xml.etree.ElementTree as ET
import numpy as np


class GetXmlGtBoxes:
    def __init__(self, keep_difficult=False, name_dict={'﻿火': '火'}):
        self.keep_difficult = keep_difficult
        self.name_dict = name_dict

    def get_groundtruth(self, _annopath):
        anno = ET.parse(_annopath).getroot()
        anno = self._preprocess_annotation(anno)
        return anno

    def _preprocess_annotation(self, target):
        gt = []
        for obj in target.iter("object"):
            difficult = int(obj.find("difficult").text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.strip()
            if name in self.name_dict:
                name = self.name_dict[name]
            bbox = obj.find("bndbox")
            bbox = [int(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            output = [name]
            output.extend(bbox)
            gt.append(output)
        return gt


def get_groundtruth(_annopath):
    anno = ET.parse(_annopath).getroot()
    anno = _preprocess_annotation(anno)
    return anno


def _preprocess_annotation(target):
    gt = []

    for obj in target.iter("object"):
        difficult = int(obj.find("difficult").text) == 1
        if not keep_difficult and difficult:
            continue
        name = obj.find("name").text.lower().strip()
        bbox = obj.find("bndbox")
        bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
        bbox[0] -= 1.0
        bbox[1] -= 1.0

        output = {'position': bbox, 'label': name_dict[name],
                  'score': (random.choice([8, 9]) + random.random()) * 0.1}

        gt.append(output)

    size = target.find("size")
    return gt
