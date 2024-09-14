import cv2
from lgdet.util.util_data_aug import Dataaug
import numpy as np
import torch
import random
import math
from torchvision import transforms


class LgTransformer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataaug = Dataaug(cfg)

    def aug_mosaic(self, imglabs, s=640):
        # loads images in a mosaic

        labels4 = []
        xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center x, y

        for i, imglab in enumerate(imglabs):
            # Load image
            img, lab, info = imglab
            h0, w0, c = img.shape
            r = s / max(h0, w0)  # resize image to img_size
            if r != 1:  # always resize down, only resize up if training with augmentation
                img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
            h, w, c = img.shape
            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            pad_w = x1a - x1b
            pad_h = y1a - y1b

            # Labels
            labels = lab.copy()
            if lab.size > 0:  # Normalized xywh to pixel xyxy format
                try:
                    labels[:, 1] = r * lab[:, 1] + pad_w
                    labels[:, 2] = r * lab[:, 2] + pad_h
                    labels[:, 3] = r * lab[:, 3] + pad_w
                    labels[:, 4] = r * lab[:, 4] + pad_h
                except:
                    a = 0
            labels4.append(labels)

        # Concat/clip labels
        if len(labels4):
            labels4 = np.concatenate(labels4, 0)
            # np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop
            np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_affine

        # Augment
        # img4 = img4[s // 2: int(s * 1.5), s // 2:int(s * 1.5)]  # center crop (WARNING, requires box pruning)
        img4, labels4 = self.random_affine(img4, labels4, degrees=1.98, translate=0.05, scale=0.05, shear=0.64, border=-s // 2)  # border to remove
        return img4, labels4

    def random_affine(self, img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, border=0):
        # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
        # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
        # targets = [cls, xyxy]

        height = img.shape[0] + border * 2
        width = img.shape[1] + border * 2

        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-degrees, degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 - scale, 1 + scale)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

        # Translation
        T = np.eye(3)
        T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  # x translation (pixels)
        T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  # y translation (pixels)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

        # Combined rotation matrix
        M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
        if (border != 0) or (M != np.eye(3)).any():  # image changed
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

        # Transform label coordinates
        n = len(targets)
        if n:
            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = (xy @ M.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # # apply angle-based reduction of bounding boxes
            # radians = a * math.pi / 180
            # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
            # x = (xy[:, 2] + xy[:, 0]) / 2
            # y = (xy[:, 3] + xy[:, 1]) / 2
            # w = (xy[:, 2] - xy[:, 0]) * reduction
            # h = (xy[:, 3] - xy[:, 1]) * reduction
            # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

            # reject warped points outside of image
            xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
            xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            area0 = (targets[:, 3] - targets[:, 1]) * (targets[:, 4] - targets[:, 2])
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
            i = (w > 2) & (h > 2) & (area / (area0 * s + 1e-16) > 0.01) & (ar < 20)

            targets = targets[i]
            targets[:, 1:5] = xy[i]

        return img, targets

    def letter_box(self, img, label, data_info={}, new_shape=(416, 416), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
        # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        pad_w, pad_h = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            pad_w, pad_h = np.mod(pad_w, 64), np.mod(pad_h, 64)  # wh padding
        elif scaleFill:  # stretch
            pad_w, pad_h = 0.0, 0.0
            new_unpad = new_shape
            ratio = [new_shape[0] / shape[1], new_shape[1] / shape[0]]  # width, height ratios

        pad_w /= 2  # divide padding into 2 sides
        pad_h /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        if label != []:
            labels = label.copy()
            labels[:, 1] = ratio[0] * label[:, 1] + pad_w
            labels[:, 2] = ratio[1] * label[:, 2] + pad_h
            labels[:, 3] = ratio[0] * label[:, 3] + pad_w
            labels[:, 4] = ratio[1] * label[:, 4] + pad_h
        else:
            labels = label
        data_info['img_raw_size(h,w)'] = shape
        data_info['ratio(w,h)'] = np.asarray(ratio)  # new/old
        data_info['padding(w,h)'] = np.asarray([pad_w, pad_h])
        data_info['size_now(h,w)'] = np.asarray(img.shape[:2])

        return img, labels, data_info

    def decode_pad2size(self, pre_labels, data_infos):
        '''

        :param pre_labels: relative labels
        :param data_infos:
        :return:  absolute labels.
        '''

        pre_labels_out = []
        for i, data_info in enumerate(data_infos):
            if len(pre_labels[i]) == 0:
                pre_labels_out.append([])
            else:
                if pre_labels[i].is_cuda == True:
                    pre_label = pre_labels[i].cpu()
                else:
                    pre_label = pre_labels[i]
                pre_label = pre_label.detach().numpy()
                ratio = data_info['ratio(w,h)']  # new/old
                pad_w, pad_h = data_info['padding(w,h)']
                if self.cfg.TRAIN.RELATIVE_LABELS:
                    shapeh, shapew = data_info['size_now(h,w)']
                else:
                    shapeh, shapew = 1, 1
                labels = pre_label.copy()
                labels[:, 2] = (pre_label[:, 2] * shapew - pad_w) / ratio[0]  # absolute labels.
                labels[:, 3] = (pre_label[:, 3] * shapeh - pad_h) / ratio[1]
                labels[:, 4] = (pre_label[:, 4] * shapew - pad_w) / ratio[0]
                labels[:, 5] = (pre_label[:, 5] * shapeh - pad_h) / ratio[1]
                pre_labels_out.append(labels)
        return pre_labels_out

    def augment_hsv(self, im, label, data_info, hgain=0.05, sgain=0.5, vgain=0.5):
        # HSV color-space augmentation
        if hgain or sgain or vgain:
            r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
            hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
            dtype = im.dtype  # uint8

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed
        return im, label, data_info

    def data_aug(self, img, label, data_info):
        labels = []
        try_tims = 0
        while [] in labels or len(labels) == 0:
            imgs, labels = self.dataaug._augmenting(aug_way_ids=([5, 6, 10, 11, 12, 13, 14, 15, 16, 20, 22], [20, 25]),
                                                    datas=([img], [label]))  # [11,20, 21, 22]
            try_tims += 1
            if try_tims > 100:
                imgs = [img]
                labels = [label]
        img_after = imgs[0]
        label_after = labels[0]
        label_after = np.asarray(label_after)
        return img_after, label_after, data_info

    def data_aug_classification(self, img, data_info):
        imgs, labels = self.dataaug._augmenting(aug_way_ids=([5, 6, 10, 11, 12, 13, 14, 15, 16, 20, 22], [20, 25]),
                                                datas=([img], None))  # [11,20, 21, 22]

        img_after = imgs[0]
        return img_after, data_info

    def resize(self, img, label, size, data_info):
        '''

        :param img:
        :param label: [cls, x1, y1, x2, y2]
        :param size:
        :param data_info:
        :return:
        '''

        img_after = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)
        img_size = img.shape
        ratio = np.asarray([size[1] / img_size[1], size[0] / img_size[0]])  # (W,H)
        label = np.asarray(label)
        label[:, [1, 3]] = label[:, [1, 3]] * ratio[0]
        label[:, [2, 4]] = label[:, [2, 4]] * ratio[1]

        data_info['ratio(w,h)'] = ratio * data_info['ratio(w,h)']
        data_info['padding(w,h)'] = ratio * data_info['padding(w,h)']
        data_info['size_now(h,w)'] = img_after.shape[:2]
        return img_after, label, data_info

    def resize_max_min_size(self, img, label, data_info, input_ksize=[800, 1333]):
        '''
        resize img and blabel
        Returns
        img_paded: input_ksize
        blabel: [None,4]
        '''
        min_side, max_side = input_ksize
        h, w, _ = img.shape

        smallest_side = min(w, h)
        largest_side = max(w, h)
        scale = min_side / smallest_side
        if largest_side * scale > max_side:
            scale = max_side / largest_side
        nw, nh = int(scale * w), int(scale * h)
        img_resized = cv2.resize(img, (nw, nh))

        pad_w = 32 - nw % 32
        pad_h = 32 - nh % 32

        pad_w /= 2  # divide padding into 2 sides
        pad_h /= 2

        # img_paded = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)
        # img_paded[:nh, :nw, :] = img_resized

        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        img_paded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # add border

        data_info['ratio(w,h)'] = (scale, scale)
        data_info['padding(w,h)'] = (pad_w, pad_h)
        data_info['size_now(h,w)'] = img_paded.shape[:2]
        if label is None:
            pass
        else:
            label[:, [1, 3]] = label[:, [1, 3]] * scale + pad_w
            label[:, [2, 4]] = label[:, [2, 4]] * scale + pad_h

        return img_paded, label, data_info

    def transpose(self, img, label=None):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, dtype=np.float32) / 255.
        # img = (img - self.cfg.mean) / self.cfg.std
        img = torch.from_numpy(img).permute(2, 0, 1)  # C, H, W
        img = transforms.Normalize(self.cfg.mean, self.cfg.std, inplace=True)(img)
        if label is None:
            return img, label

        if isinstance(label, list):
            label = np.asarray(label, np.float32)

        label = np.insert(label, 0, 0, 1)
        label = torch.from_numpy(label).to(torch.float32)
        return img, label

    def relative_label(self, img, label):
        # RELATIVE LABELS:# x1y1x2y2
        if label is None:
            return img, label

        h, w, c = img.shape
        if isinstance(label, list):
            label = np.asarray(label, np.float32)
        mask = np.asarray([1, w, h, w, h], np.float32)
        label = label / mask
        return img, label

    def img_binary(self, img, label, data_info):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('img', img)
        # cv2.waitKey()
        label = np.asarray(label)
        use_txt_scal_img = 0
        if use_txt_scal_img:
            txt = label[label[:, 0] == 33]
            if txt != []:
                w_txt = min((txt[:, 3] - txt[:, 1]).mean(), (txt[:, 4] - txt[:, 2]).mean())
                ratio = 20 / w_txt
                newsize = int(img.shape[0] * ratio), int(img.shape[1] * ratio)
                img, label, data_info = self.resize(img, label, newsize, data_info)

        img, binary = cv2.threshold(img, 210, 255, cv2.THRESH_BINARY)
        img = np.stack((binary,) * 3, axis=-1)
        # cv2.imshow('img', img)
        # cv2.waitKey()
        return img, label, data_info
    #
    # def rescale_size(self, old_size, scale, return_scale=False):
    #     """Calculate the new size to be rescaled to."""
    #
    #     w, h = old_size
    #
    #     max_long_edge = max(scale)
    #     max_short_edge = min(scale)
    #     scale_factor = min(max_long_edge / max(h, w),
    #                        max_short_edge / min(h, w))
    #
    #     new_size = self._scale_size((w, h), scale_factor)
    #
    #     return new_size, scale_factor

    # def letter_box_lg(self, img, label, size, decode=False):
    #     h, w, c = img.shape
    #     ratio_w = size[1] / w
    #     ratio_h = size[0] / h
    #     # if ratio_w
    #     if ratio_w < ratio_h:
    #         ratio_min = ratio_w
    #     else:
    #         ratio_min = ratio_h
    #
    #     # resize:
    #
    #     img = cv2.resize(img, None, fx=ratio_min, fy=ratio_min)
    #     h, w, c = img.shape
    #     # Determine padding
    #     # pad =[left, right, top, bottom]
    #     if abs(w - size[1]) > abs(h - size[0]):
    #         dim_diff = abs(w - size[1])
    #         pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    #         pad = (pad1, pad2, 0, 0)
    #     else:
    #         dim_diff = abs(h - size[0])
    #         pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    #         pad = (0, 0, pad1, pad2)
    #
    #     # Add padding
    #     img = cv2.copyMakeBorder(img, pad[2], pad[3], pad[0], pad[1], cv2.BORDER_CONSTANT, value=[0, 0, 0])
    #     if not decode:
    #         label_after = [[
    #             lab[0],
    #             lab[1] * ratio_min + pad[0],
    #             lab[2] * ratio_min + pad[2],
    #             lab[3] * ratio_min + pad[1],
    #             lab[4] * ratio_min + pad[3]
    #         ] for lab in label]
    #     else:
    #         label_after = [[
    #             lab[0],
    #             (lab[1] - pad[0]) / ratio_min,
    #             (lab[2] - pad[2]) / ratio_min,
    #             (lab[3] - pad[1]) / ratio_min,
    #             (lab[4] - pad[3]) / ratio_min,
    #         ] for lab in label]
    #
    #     return img, label_after

    # def _scale_size(self, size, scale):
    #     """Rescale a size by a ratio.
    #
    #     Args:
    #         size (tuple[int]): (w, h).
    #         scale (float): Scaling factor.
    #
    #     Returns:
    #         tuple[int]: scaled size.
    #     """
    #     w, h = size
    #     return int(w * float(scale) + 0.5), int(h * float(scale) + 0.5)

    # def imnormalize(self, img, mean, std, to_rgb=True):
    #     """Inplace normalize an image with mean and std.
    #
    #     Args:
    #         img (ndarray): Image to be normalized.
    #         mean (ndarray): The mean to be used for normalize.
    #         std (ndarray): The std to be used for normalize.
    #         to_rgb (bool): Whether to convert to rgb.
    #
    #     Returns:
    #         ndarray: The normalized image.
    #     """
    #     # cv2 inplace normalization does not accept uint8
    #     assert img.dtype != np.uint8
    #     mean = np.float64(mean.reshape(1, -1))
    #     stdinv = 1. / np.float64(std.reshape(1, -1))
    #     if to_rgb:
    #         cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
    #     cv2.subtract(img, mean, img)  # inplace
    #     cv2.multiply(img, stdinv, img)  # inplace
    #     return img
    #
    # def imdenormalize(self, img, mean, std, to_bgr=True):
    #     assert img.dtype != np.uint8
    #     mean = mean.reshape(1, -1).astype(np.float64)
    #     std = std.reshape(1, -1).astype(np.float64)
    #     img = cv2.multiply(img, std)  # make a copy
    #     cv2.add(img, mean, img)  # inplace
    #     if to_bgr:
    #         cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)  # inplace
    #     img = img.astype(np.uint8)
    #     return img
