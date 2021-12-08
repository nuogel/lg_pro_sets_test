# copy smale target with bounding boxes to background pics.
import json
import os
import cv2
import random
import numpy as np
import tqdm
import glob


class CopyLittleTarget:
    def __init__(self, bg_imgs_path, bg_labs_path, source_imgs_path, source_labs_path, copy_number, save_path):
        self.bg_imgs_path = bg_imgs_path
        self.bg_labs_path = bg_labs_path
        self.source_imgs_path = source_imgs_path
        self.source_labs_path = source_labs_path
        self.bg_imgs_list = os.listdir(self.bg_imgs_path)
        self.targets_dict = {'文字': [],
                             '打包垃圾': []}
        self.copy_number = copy_number
        for cls, img_p, lab_p in zip(('文字', '打包垃圾'), source_imgs_path, source_labs_path):
            for img in os.listdir(img_p):
                name = os.path.basename(img)
                img_ = os.path.join(img_p, name)
                if lab_p is not None:
                    lab_ = os.path.join(lab_p, name.replace('jpg', 'json'))
                else:
                    lab_ = None  # name.replace('.jpg', '').split('_')[-1]
                self.targets_dict[cls].append([img_, lab_])

        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'labels'), exist_ok=True)

    def run(self):
        for ii, img_name in enumerate(self.bg_imgs_list):
            print('deeling with pic:', ii, '-', img_name)

            dest_img, dest_mask = self._make_dest_images(img_name, self.targets_dict)

            dest_mask[dest_mask != 0] = 255
            cv2.imshow('img', dest_img)
            cv2.waitKey(1)
            # cv2.imshow('img', dest_mask)
            # cv2.waitKey()

            # dest_lab_path = os.path.join(self.save_path, 'labels', img_name.replace('.jpg', '.png'))
            dest_img_path = os.path.join(self.save_path, 'images', img_name)
            cv2.imwrite(dest_img_path, dest_img)
            # cv2.imwrite(dest_lab_path, dest_mask)
            json_label = False
            txt_label = True
            if txt_label == True:
                dest_lab_path = os.path.join(self.save_path, 'labels', img_name.replace('.jpg', '.txt'))
                kernel = np.ones((3, 3), np.uint8)
                dest_mask = cv2.dilate(dest_mask, kernel, iterations=1)
                contours, _ = cv2.findContours(dest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # print('ctrs:', len(contours))
                contours = self._filter_area(contours, min_area=5)
                # TODO: add type:out and in
                position_list = []
                txt = ''
                def fun(x):
                    return str(int(x))
                for contour in contours:
                    (x, y), (w, h), theta = cv2.minAreaRect(contour)

                    txt += ','.join(list(map(fun,[x, y, x + w, y, x + w, y + w, x, y + w])))
                    txt += ',label' + '\n'
                f = open(dest_lab_path, 'w')
                f.write(txt)
                f.close()

            if json_label:
                '''
                [{"positions": [{"type": "out", "point": [{"x": 728, "y": 512}, {"x": 979, "y": 512}, {"x": 979, "y": 796}, {"x": 728, "y": 796}]}, {"type": "in", "point": [{"x": 730, "y": 520}, {"x": 800, "y": 520}, {"x": 800, "y": 600}, {"x": 730, "y": 600}]}, {"type": "in", "point": [{"x": 900, "y": 600}, {"x": 950, "y": 600}, {"x": 950, "y": 750}, {"x": 900, "y": 750}]}], "labels": {"实体类型": "轿车", "属性": {"品牌": "本田锋范", "年份": "2013-2017", "车身颜色": "红色"}}}]
                '''

                kernel = np.ones((3, 3), np.uint8)
                dest_mask = cv2.dilate(dest_mask, kernel, iterations=1)
                contours, _ = cv2.findContours(dest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # print('ctrs:', len(contours))
                contours = self._filter_area(contours, min_area=5)
                # TODO: add type:out and in
                position_list = []
                for contour in contours:
                    contour = np.squeeze(contour, axis=1)
                    point_list = []
                    for point in contour:
                        point_list.append({'x': point[0], 'y': point[1]})
                    point_dict = {'type': 'out',
                                  'point': point_list}
                    position_list.append(point_dict)
                json_dict = {'positions': position_list,
                             'labels': {'实体类型': '垃圾'}}
                # jf = json.dumps(json_dict)
                dest_lab_path = os.path.join(self.save_path, 'labels', img_name.replace('.jpg', '.json'))
                json.dump(json_dict, dest_lab_path)

    def _filter_area(self, ctrs, min_area=64):
        ret = list()
        for i in range(0, len(ctrs)):
            area = cv2.contourArea(ctrs[i])
            if area > min_area:
                ret.append(ctrs[i])
        return ret

    def _read_json_mask(self, file):
        f = open(file)
        source_bboxes = json.load(f)
        labels = []
        for annoinfo in source_bboxes['annotateInfo']:
            label = {}
            name = annoinfo['entityDetail']['name']
            positions = json.loads(annoinfo['positions'][0]['positions'])['meaningful']
            points = []
            for position in positions:
                points.append([int(position['x']), int(position['y'])])
            label['name'] = name
            label['points'] = points
            labels.append(label)
        return labels

    def _show_mask(self, img, label):
        for bblines in label:
            contours = np.asarray(bblines['points'], dtype=np.int)
            img = cv2.polylines(img, [contours], True, (0, 255, 255), 3)
        cv2.imshow('img', img)
        cv2.waitKey()
        cv2.imwrite(str(random.randint(0, 1000)) + '.jpg', img)

    def _read_source_targets(self, mask_img_path, mask_lab_path):  # little target with its bounding box as a pic size.
        mask_img = cv2.imread(mask_img_path)
        if mask_lab_path is not None:
            mask = np.zeros(mask_img.shape[:2], dtype=np.uint8)
            mask_points = self._read_json_mask(mask_lab_path)
            # self._show_mask(mask_img, mask_points)

            for mask_point in mask_points:
                if mask_point['name'] == '文字':
                    mask = cv2.fillPoly(mask, [np.asarray(mask_point['points'])], 255)
            # cv2.imshow('mask', mask)
            # cv2.waitKey()
        else:
            mask = np.ones(mask_img.shape[:2], dtype=np.uint8) * 255

        return mask_img, mask

    def _make_dest_images(self, dest_img, targets_dict):
        dest_img_path = os.path.join(self.bg_imgs_path, dest_img)
        if self.bg_labs_path:
            dest_lab_path = os.path.join(self.bg_labs_path, dest_img.replace('.jpg', '.json'))
        else:
            dest_lab_path = self.bg_labs_path
        dest_img, dest_mask = self._read_source_targets(dest_img_path, dest_lab_path)
        d_size = dest_img.shape
        if self.copy_number <= 0:
            return dest_img, dest_mask

        for k, v in targets_dict.items():
            if len(v) >= self.copy_number:
                s_targets = random.sample(v, self.copy_number)
            else:
                s_targets = v
            for s_target_p in s_targets:
                s_img, s_mask = self._read_source_targets(s_target_p[0], s_target_p[1])
                # max_hw = random.randint(50, 150)
                # if s_img.shape[0] > max_hw or s_img.shape[1] > max_hw:
                #     s_img = cv2.resize(s_img, (max_hw, max_hw))
                #     s_mask = cv2.resize(s_mask, (max_hw, max_hw))
                s_size = s_img.shape
                x1 = random.randint(0, d_size[1] - s_size[1])
                y1 = random.randint(0, d_size[0] - s_size[0])

                d_box = dest_mask[y1:y1 + s_size[0], x1:x1 + s_size[1]]
                if d_box.sum() == 0:
                    s_mask_inv = cv2.bitwise_not(s_mask)
                    s_img = cv2.bitwise_and(s_img, s_img, mask=s_mask)

                    d_img = dest_img[y1:y1 + s_size[0], x1:x1 + s_size[1]]
                    d_img = cv2.bitwise_and(d_img, d_img, mask=s_mask_inv)

                    dst = cv2.add(s_img, d_img)  # 相加即可

                    dest_img[y1:y1 + s_size[0], x1:x1 + s_size[1]] = dst
                    dest_mask[y1:y1 + s_size[0], x1:x1 + s_size[1]] = s_mask

        return dest_img, dest_mask

    def crop_source_images(self):
        imgpath = '/media/dell/data/ocr/计费清单识别/集团-计费清单/images'
        imgcroppath = '/media/dell/data/ocr/计费清单识别/集团-计费清单/crop'
        labpath = '/media/dell/data/ocr/计费清单识别/集团-计费清单/labels'
        for img_i in os.listdir(imgpath):
            imgp = os.path.join(imgpath, img_i)
            img = cv2.imread(imgp)
            labp = os.path.join(labpath, img_i.split('.')[0] + '.txt')
            f = open(labp, 'r')
            for i, line in enumerate(f.readlines()):
                tmp = line.split(',')
                loc_length = 8
                loc = tmp[:loc_length]
                poly = np.asarray(list(map(float, loc))).reshape((-1, 2))
                minx = int(poly[:, 0].min())
                maxx = int(poly[:, 0].max())
                miny = int(poly[:, 1].min())
                maxy = int(poly[:, 1].max())
                imgcrop = img[miny:maxy, minx:maxx, :]
                cropsavename = os.path.join(imgcroppath, img_i.split('.')[0] + '_' + str(i) + '_' + str(tmp[-1].strip()) + '.jpg')
                cv2.imwrite(cropsavename, imgcrop)


if __name__ == '__main__':
    bg_imgs_path = '/media/dell/data/ocr/计费清单识别/集团-计费清单/images'
    bg_labs_path = '/media/dell/data/ocr/计费清单识别/集团-计费清单/labels_plate'  # None  #
    save_path = '/media/dell/data/ocr/计费清单识别/集团-计费清单/little_targets'

    source_imgs_paths = ['/media/dell/data/ocr/计费清单识别/集团-计费清单/crop']
    source_labs_paths = [None]
    copy_number = 50
    clt = CopyLittleTarget(bg_imgs_path, bg_labs_path, source_imgs_paths, source_labs_paths, copy_number, save_path)
    clt.run()