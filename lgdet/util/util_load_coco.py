import os.path as osp
import tempfile

import mmcv
import numpy as np
from pycocotools.coco import COCO
from torch.utils.data import DataLoader


class COCODataset:
    def __init__(self, ann_file, cfg=None):
        self.cfg = cfg
        self.ann_file = ann_file
        self._load_annotations(self.ann_file)

    def _load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco.getImgIds()

    def _set_group_flag(self):  # LG: this is important
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.data_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def prepare_data(self, file_name):
        imgid = int(file_name.split('.')[0].strip())
        img_info = self.coco.loadImgs([imgid])[0]
        ann_info = self.get_ann_info(img_info)
        return img_info, ann_info

    def get_ann_info(self, img_info):
        img_id = img_info['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        ann_info_out = self._parse_ann_info(img_info, ann_info)
        return ann_info_out

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cfg.TRAIN.CLASSES[self.cat2label[ann['category_id']]])
                gt_masks_ann.append(ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            # gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = []

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['file_name'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann


if __name__ == '__main__':
    dataset = COCODataset('/media/lg/2628737E28734C35/coco/annotations/instances_train2017.json')
    traindata = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    # data = iter(traindata)
    # i, l = next(data)
    # or use the next code.
    for i, (img, lab) in enumerate(traindata):
        print(lab)
