import os
import numpy as np
import xml.etree.ElementTree as ET
import cfg.config as cfg

CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
index_map = dict(zip(CLASSES, range(len(CLASSES))))


def validate_label(xmin, ymin, xmax, ymax, width, height):
    """Validate labels."""
    assert 0 <= xmin < width, "xmin must in [0, {}), given {}".format(width, xmin)
    assert 0 <= ymin < height, "ymin must in [0, {}), given {}".format(height, ymin)
    assert xmin < xmax <= width, "xmax must in (xmin, {}], given {}".format(width, xmax)
    assert ymin < ymax <= height, "ymax must in (ymin, {}], given {}".format(height, ymax)


def decode_xml(anno_path):
    '''
    Decode the xml to label.
    :param anno_path: xml_path
    :return: labels：[[x1, y1, x2, y2, class, hard], ...,[...]]
    '''
    tree = ET.parse(anno_path)
    root = tree.getroot()
    label = []
    size = root.find('size')
    width = float(size.find('width').text)
    height = float(size.find('height').text)
    for obj in root.iter('object'):
        difficult = int(obj.find('difficult').text)
        cls_name = obj.find('name').text.strip().lower()
        if cls_name not in CLASSES:
            continue
        cls_id = index_map[cls_name]

        xml_box = obj.find('bndbox')
        xmin = (float(xml_box.find('xmin').text) - 1)
        ymin = (float(xml_box.find('ymin').text) - 1)
        xmax = (float(xml_box.find('xmax').text) - 1)
        ymax = (float(xml_box.find('ymax').text) - 1)

        try:
            validate_label(xmin, ymin, xmax, ymax, width, height)
        except AssertionError as e:
            raise RuntimeError("Invalid label at {}, {}".format(anno_path, e))
        label.append([cls_id, xmin, ymin, xmax, ymax, difficult])

    label = np.array(label)
    return label


anno_path = os.path.join(cfg.LAB_PATH_VOC2007, '000005.xml')
label = decode_xml(anno_path)
print(label)
