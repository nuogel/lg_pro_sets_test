import numpy as np
import zipfile
import cv2
from util import _show_img
# sys.path.append('E:\datasets\coco\cocoapi-master\cocoapi-master\PythonAPI')  # 将你的 `pycocotools` 所在路径添加到系统环境
from pycocotools.coco import COCO  # 载入 cocoz


def _read_label_coco(data_root, split_name, catgorys=['person', 'car', ]):
    print('start read zip file')
    Z = zipfile.ZipFile(f'{data_root}/%s2017.zip' % split_name)
    print('finish read zip file')

    json_name = 'instances_%s2017.json' % split_name
    annFile = '{}//annotations_trainval2017//annotations//{}'.format(data_root, json_name)
    # initialize COCO api for instance annotations
    coco = COCO(annFile)

    catIds = coco.getCatIds(catNms=catgorys)  # get the cat's IDS
    imgIds = coco.getImgIds(catIds=catIds, )  # use the ID above to get image's ID that have these ID's instance
    imgs = coco.loadImgs(imgIds)  # use the img IDs to get the images information
    print('there are/is %d instance in this dataset' % len(imgs))
    bboxes = []
    images = []
    for img in imgs:
        img_name = '%s2017/' % split_name + img['file_name']
        print('parsing img:', img_name)
        img_b = Z.read(img_name)
        img_flatten = np.frombuffer(img_b, 'B')
        img_cv = cv2.imdecode(img_flatten, cv2.IMREAD_ANYCOLOR)
        # cv2.imshow('img', img_cv)
        # cv2.waitKey()
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)  # catIds=catIds,
        anns = coco.loadAnns(annIds)
        _bboxes = []
        for ann in anns:
            bbox = ann['bbox']
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]
            bbox = [x1, y1, x2, y2]
            cat_id = ann['category_id']
            cat_name = coco.loadCats(ids=cat_id)[0]['name']
            _bboxes.append([cat_name, bbox])
        bboxes.append(_bboxes)
        images.append(img_cv)
    images = np.asarray(images)
    return images, bboxes


if __name__ == '__main__':
    data_root = 'E://datasets//coco//lg_coco//'
    split_name = 'val'
    catgorys = ['person', 'car']
    images, bboxes = _read_label_coco(data_root, split_name, catgorys)
    _show_img(images, bboxes)
