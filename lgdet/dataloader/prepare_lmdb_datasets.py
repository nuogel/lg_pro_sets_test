import lmdb
import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import tqdm
import pickle


def load_labe_xml(path):
    label_infos = {
        'cls_names': [],
        'bboxes': []
    }
    if os.path.basename(path).split('.')[-1] == 'xml':
        tree = ET.parse(path)
        root = tree.getroot()
        size = root.find('size')
        height = int(size.find('height').text)
        width = int(size.find('width').text)
        depth = int(size.find('depth').text)

        label_infos['img_size_hwc'] = [height, width, depth]
        for obj in root.findall('object'):
            label_infos['cls_names'].append(obj.find('name').text)
            bbox = obj.find('bndbox')
            box_x1 = float(bbox.find('xmin').text)
            box_y1 = float(bbox.find('ymin').text)
            box_x2 = float(bbox.find('xmax').text)
            box_y2 = float(bbox.find('ymax').text)
            label_infos['bboxes'].append([box_x1, box_y1, box_x2, box_y2])
    return label_infos


def lmdb_writer(lmdb_path):
    path_base = '/media/lg/2628737E28734C35/coco/'
    txt_base = '/media/lg/SSD_WorkSpace/LG/GitHub/lg_pro_sets/datasets/OBD_idx_stores/COCO/COCO_{}.txt'
    model = 'train'
    txt_path = txt_base.format(model)
    lines = open(txt_path).readlines()

    env = lmdb.Environment(lmdb_path, subdir=True,
                           map_size=int(1e13), max_dbs=2, lock=False)

    db_image = env.open_db('image'.encode(), create=True)
    db_label = env.open_db('label'.encode(), create=True)

    for line in tqdm.tqdm(lines):
        tmp = line.split('┣┫')
        img_name = tmp[0].strip()
        img_path = os.path.join(path_base, tmp[1].strip())
        lab_path = os.path.join(path_base, tmp[2].strip())

        label_infos = load_labe_xml(lab_path)
        label_infos['img_name'] = img_name
        label_infos['img_path'] = img_path
        label_infos['lab_path'] = lab_path

        with open(img_path, 'rb') as f:
            # 读取图像文件的二进制格式数据
            image_bin = f.read()

        with env.begin(write=True) as image_writer:
            image_writer.put(img_name.encode(), image_bin, db=db_image)

        with env.begin(write=True) as lmdb_writer:
            lmdb_writer.put(img_name.encode(),
                            pickle.dumps(label_infos), db=db_label)

    env.close()
    print('lmdb finished')


def lmdb_reader(lmdb_path):
    txt_base = '/media/lg/SSD_WorkSpace/LG/GitHub/lg_pro_sets/datasets/OBD_idx_stores/COCO/COCO_{}.txt'
    model = 'test'
    txt_path = txt_base.format(model)
    lines = open(txt_path).readlines()

    env = lmdb.open(lmdb_path, max_dbs=2, lock=False)

    db_image = env.open_db('image'.encode())
    db_label = env.open_db('label'.encode())

    txn_image = env.begin(write=False, db=db_image)
    txn_label = env.begin(write=False, db=db_label)

    for line in lines:
        tmp = line.split('┣┫')
        img_name = tmp[0].strip()

        # 获取图像数据
        image_bin = txn_image.get(img_name.encode())
        label_bin = txn_label.get(img_name.encode())

        # 将二进制文件转为十进制文件（一维数组）
        image_buf = np.frombuffer(image_bin, dtype=np.uint8)
        # 将数据转换(解码)成图像格式
        # cv2.IMREAD_GRAYSCALE为灰度图，cv2.IMREAD_COLOR为彩色图
        img = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
        cv2.imshow('img', img)
        cv2.waitKey()

        label = pickle.loads(label_bin)
        print(label)


if __name__ == '__main__':
    lmdb_path = '/media/lg/2628737E28734C35/coco/train2017_lmdb'

    lmdb_writer(lmdb_path)
    lmdb_reader(lmdb_path)
