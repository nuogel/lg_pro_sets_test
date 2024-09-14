# -*- coding: utf-8 -*-
"""
eva_sys - 当前项目的名称。
util_show_img - 读取图片数据，对应标签数据，将标注框画到图片上，显示出来

"""
import random

import cv2
import os
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def get_ALL_File(dir_path, seq=['.png']):
    file_list = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            filehead, ext = os.path.splitext(file)
            if ext in seq:
                filename = os.path.join(root, file)
                file_list.append(filename)

    return file_list


def _read_line(path, labelnames):
    objs = []
    if os.path.basename(path).split('.')[-1] == 'txt':
        file_open = open(path, 'r', encoding='utf-8')
        for line in file_open.readlines():
            if ";" in line.strip():
                tmps = []
                tmps.append(line.strip().split(";")[0])
                tmps.extend(line.strip().split(";")[1].split(' '))
            else:
                tmps = line.strip().split(' ')
            # print(len(tmps))
            # print(tmps)
            name = tmps[0]
            score = 1.0
            if len(tmps) == 5:
                box_x1 = float(tmps[1])
                box_y1 = float(tmps[2])
                box_x2 = float(tmps[3])
                box_y2 = float(tmps[4])

                if box_x1 < 1 and box_y1 < 1 and box_x2 < 1 and box_y2 < 1:
                    print('use yolo bboxes type')

                    img_w = 1920
                    img_h = 1080

                    x = box_x1 * img_w
                    y = box_y1 * img_h
                    w = box_x2 * img_w
                    h = box_y2 * img_h

                    box_x1 = x - w / 2.
                    box_y1 = y - h / 2.
                    box_x2 = x + w / 2.
                    box_y2 = y + h / 2.
            elif len(tmps) == 6:
                # box_x1 = float(tmps[1])
                # box_y1 = float(tmps[2])
                # box_x2 = float(tmps[3])
                # box_y2 = float(tmps[4])
                # plant_number = tmps[5]

                score = float(tmps[1])
                box_x1 = float(tmps[2])
                box_y1 = float(tmps[3])
                box_x2 = float(tmps[4])
                box_y2 = float(tmps[5])

                if box_x1 < 1 and box_y1 < 1 and box_x2 < 1 and box_y2 < 1:
                    print('use yolo bboxes type')

                    img_w = 1920
                    img_h = 1080

                    x = box_x1 * img_w
                    y = box_y1 * img_h
                    w = box_x2 * img_w
                    h = box_y2 * img_h

                    box_x1 = x - w / 2.
                    box_y1 = y - h / 2.
                    box_x2 = x + w / 2.
                    box_y2 = y + h / 2.

            else:
                # box_x1 = float(tmps[4])
                # box_y1 = float(tmps[5])
                # box_x2 = float(tmps[6])
                # box_y2 = float(tmps[7])
                score = float(tmps[1])

                box_x1 = float(tmps[2])
                box_y1 = float(tmps[3])
                box_x2 = float(tmps[4])
                box_y2 = float(tmps[5])
            # if name in CLASS:
            objs.append([name, score, str(box_x1), str(box_y1), str(box_x2), str(box_y2)])  # + ': ' + '%.3f' % score
    elif os.path.basename(path).split('.')[-1] == 'xml':
        tree = ET.parse(path)
        # 　label_path = fpath.replace("images", "labels")
        # im_file = in_file[:-4].replace("labels", "images") + ".jpg"
        # img = cv2.imread(im_file)
        root = tree.getroot()
        # bndboxlist = []
        for object in root.findall('object'):  # 找到root节点下的所有country节点
            bndbox = object.find('bndbox')  # 子节点下节点rank的值
            name = object.find('name').text
            xmin = float(bndbox.find('xmin').text)
            xmax = float(bndbox.find('xmax').text)
            ymin = float(bndbox.find('ymin').text)
            ymax = float(bndbox.find('ymax').text)
            # if name in CLASS:
            objs.append([name, xmin, ymin, xmax, ymax])
            if name not in labelnames:
                labelnames.append(name)
    return objs, labelnames


def _write_line(before_file, new_target, after_file):
    if os.path.basename(before_file).split('.')[-1] == 'txt':
        lines = ''
        for index in range(len(new_target)):
            new_xmin = new_target[index][0]
            new_ymin = new_target[index][1]
            new_xmax = new_target[index][2]
            new_ymax = new_target[index][3]
            name = new_target[index][4]
            line = name + " " + '%.4f' % new_xmin + " " + '%.4f' % new_ymin + " " + '%.4f' % new_xmax + " " + '%.4f' % new_ymax + "\n"
            lines += line
        with open(after_file, 'w') as f:
            f.write(lines)
    elif os.path.basename(before_file).split('.')[-1] == 'xml':
        tree = ET.parse(before_file)
        elem = tree.find('filename')
        fpath, fname = os.path.split(after_file)
        elem.text = (fname[:-4] + '.jpg')
        # size = tree.find('size')
        # height, width, channel = img.shape
        # h = size.find('height')
        # h.text = str(height)
        # w = size.find('width')
        # w.text = str(width)
        # depth = size.find('depth')
        # depth.text = str(channel)

        xmlroot = tree.getroot()
        index = 0
        # print(len( xmlroot.findall('object')))

        for object in xmlroot.findall('object'):  # 找到root节点下的所有country节点
            bndbox = object.find('bndbox')  # 子节点下节点rank的值

            new_xmin = new_target[index][1]
            new_ymin = new_target[index][2]
            new_xmax = new_target[index][3]
            new_ymax = new_target[index][4]

            xmin = bndbox.find('xmin')
            xmin.text = new_xmin
            ymin = bndbox.find('ymin')
            ymin.text = new_ymin
            xmax = bndbox.find('xmax')
            xmax.text = new_xmax
            ymax = bndbox.find('ymax')
            ymax.text = new_ymax

            index = index + 1

        tree.write(after_file)


def _read_datas(im_file, lab_file, checklabelsonly=0, labelnames=[]):
    img = None
    if not checklabelsonly:
        img = cv2.imread(im_file)
        if img is not None:
            image_size = img.shape[:-1]  # the last pic as the shape of a batch.
        # images.append(img)
        else:
            print('imread img is NONE.')
    label, labelnames = _read_line(lab_file, labelnames)
    if label is None:
        print("label error")
    return img, label, labelnames


def _show_img(images, labels, show_img=True, show_time=None, save_img=False, save_video=0, save_path=None,
              video_writer=None):
    if labels:
        for _, label in enumerate(labels):
            if len(label) == 5:  # in shape of [class, x1, y1, x2, y2]
                class_out, box = label[0], label[1:5]

            elif len(label) == 2:  # [class, bbox[x1,y1,x2,y2]]
                class_out = label[0]
                box = label[1]
            elif len(label) == 6:
                class_out, score, box = label[0], label[1], label[2:]
            else:
                print('error: util_show_img-->no such a label shape')
                continue
            xmin = int(float(box[0]))
            ymin = int(float(box[1]))
            xmax = int(float(box[2]))
            ymax = int(float(box[3]))
            # text = class_out+"||"+ " ".join([str(i) for i in box])
            class_out_dict = {'person': '站立', 'fall': '跌倒'}
            color = {'default': (0, 255, 0), 'person': (0, 255, 0), 'fall': (0, 0, 255)}
            text = class_out
            # text = class_out_dict[class_out] #+ str('--%.3f' % score)
            # text = class_out_dict[class_out] + str('--%.3f' % score)
            # text = '占道经营' + str('--%.3f' % (0.8+0.1*random.random()))
            # class_out='占道经营'
            # color = {'占道经营': (0, 0, 255)}

            cv2.rectangle(images, (xmin, ymin), (xmax, ymax), color['default'], thickness=3)
            tryPIL = 1
            if label[0] in ['人脸关键点']:
                continue
            if tryPIL:
                # PIL图片上打印汉字
                pilImg = Image.fromarray(cv2.cvtColor(images, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pilImg)
                size_font = 40
                font = ImageFont.truetype(font=f'font/simhei.ttf', size=size_font,
                                          encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小
                draw.text((xmin, ymin - size_font), text, color['default'],
                          font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
                # PIL图片转cv2 图片
                images = cv2.cvtColor(np.array(pilImg), cv2.COLOR_RGB2BGR)
            else:
                cv2.putText(images, class_out, (xmin, ymin), 3, 3, (0, 255, 255))

    # cv2.putText(images, 'The Number of Cars is : %d' % len(labels), (600, 220), 1, 2, (0, 0, 255), thickness=2)
    # cv2.putText(images, 'Made by AI Team of Chengdu Fourier Electronic', (600, 250), 1, 2, (0, 0, 255), thickness=2)

    if show_img:
        cv2.imshow('img', images)
        cv2.waitKey(show_time)
    else:
        cv2.imshow('img', images)
        cv2.waitKey()
    if save_img:
        cv2.imwrite(save_path, images)
    if save_video:
        video_writer.write(images)


def check_is_file(file):
    if not os.path.isfile(file):
        print(file, 'is not found')
        return False
    else:
        return True


def main(img_fold, label_fold):
    # _show_img(img, label)
    # local_label_files2 = get_ALL_File(label_folds, [".json"])
    local_img_files = get_ALL_File(img_fold, [".jpg", '.png'])
    local_label_files = get_ALL_File(label_fold, [".txt", ".xml"])

    local_img_files.sort()
    local_label_files.sort()
    lab_num = len(local_label_files)
    img_num = len(local_img_files)
    print(lab_num, img_num)

    save_video = 0
    save_image_with_boxes = 0
    checklabelsonly = 0

    if save_video:
        base_path = os.path.join(img_folds, '../saved')
        os.makedirs(base_path, exist_ok=True)
        video_dir = os.path.join(base_path, 'video.avi')
        fps = 12
        img_size = (1920, 1080)
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        video_writer = cv2.VideoWriter(video_dir, fourcc, fps, img_size)
    else:
        video_writer = None
    labelnames = []
    pairfiles = 0
    for index in range(min(img_num, lab_num)):
        # if index <= 7800 // 5 or index > 8300 // 5: continue
        im_file = local_img_files[index]
        label_file = im_file.replace('images', 'labels').replace('.jpg', '.xml')  # local_label_files[index]
        if not check_is_file(im_file) or not check_is_file(label_file):
            continue
        if print_path:
            print(im_file, '==>>>', label_file)
        img, label, labelnames = _read_datas(im_file, label_file, checklabelsonly, labelnames)
        # if '头部' in labelnames or '通用-头部' in labelnames:
        #     return True
        # else:
        #     return  False
        pairfiles += 1
        if save_image_with_boxes:
            save_path = os.path.join(base_path, str(index) + '.jpg')
            save_image = True
        else:
            save_path = None
            save_image = False
        if not checklabelsonly:
            _show_img(img, label, show_img=True, show_time=0, save_img=save_image, save_video=save_video,
                      save_path=save_path, video_writer=video_writer)
        if save_video:
            video_writer.release()
        # if index >= 9750 - 500: break
    if save_video: video_writer.release()
    print('labelnames', labelnames)
    print('finish')


if __name__ == '__main__':
    print_path = 1

    img_folds = '/media/dell/data/personabout/person_face/人脸检测训练集/images'
    label_folds = '/media/dell/data/personabout/person_face/人脸检测训练集/labels'
    main(img_folds, label_folds)
