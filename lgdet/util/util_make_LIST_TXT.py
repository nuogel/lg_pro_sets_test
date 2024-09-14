import glob
import os
import xml.etree.ElementTree as ET


def _make_list_by_hand(path):
    list = []
    for i in range(22872):
        img1 = os.path.join(path, '%05d_img1.ppm' % i)
        img2 = os.path.join(path, '%05d_img2.ppm' % i)
        flow = os.path.join(path, '%05d_flow.flo' % i)
        if os.path.isfile(img1) and os.path.isfile(img1) and os.path.isfile(img1):
            list.append([img1, img2, flow])
    return list


def make_list(base_path, x_file, y_file):
    dir_list = []
    path_list = os.listdir(os.path.join(base_path, x_file))

    for path_i in path_list:
        x_path = os.path.join(base_path, x_file, path_i)
        y_base = path_i.split('.', -1)[0] + '.xml'
        y_path = os.path.join(base_path, y_file, y_base)
        if _is_file(x_path) and _is_file(y_path):
            # dir_list.append([path_i, x_path, y_path])
            dir_list.append([path_i, os.path.join(x_file, path_i), os.path.join(y_file, y_base)])
            print('adding:', dir_list[-1])

    return dir_list

def make_list_classfication(base_path, lab):
    dir_list = []
    for labi in lab:
        path_list = os.listdir(os.path.join(base_path, labi))
        for path_i in path_list:
            x_path = os.path.join(base_path,labi, path_i)
            if _is_file(x_path):
                dir_list.append([path_i, x_path,labi])
                print('adding:', dir_list[-1])

    return dir_list

def _is_file(path):
    if not os.path.isfile(path):
        return False

    else:

        return True
    # return False


def _label_request(path):
    # lines = open(path).readlines()
    # if lines == []:
    # for line in lines:
    #      name_dict = {'0': 'ignored regions', '1': 'pedestrian', '2': 'people',
    #                   '3': 'bicycle', '4': 'car', '5': 'van', '6': 'truck',
    #                   '7': 'tricycle', '8': 'awning-tricycle', '9': 'bus',
    #                   '10': 'motor', '11': 'others'}
    #      tmps = line.strip().split(',')
    #      realname = name_dict[tmps[5]]
    #      if realname in ['car', 'van', 'truck', 'bus']:
    #          return True
    class_name = ['Others', 'Person']
    ok_file = False
    if os.path.basename(path).split('.')[-1] == 'xml':
        tree = ET.parse(path)
        root = tree.getroot()
        AREA_OK = False
        for obj in root.findall('object'):
            cls_name = obj.find('name').text
            # bbox = obj.find('bndbox')
            # box_x1 = float(bbox.find('xmin').text)
            # box_y1 = float(bbox.find('ymin').text)
            # box_x2 = float(bbox.find('xmax').text)
            # box_y2 = float(bbox.find('ymax').text)
            # area = (box_x2-box_x1)*(box_y2-box_y1)
            if cls_name in class_name:
                ok_file = True
                return ok_file
    return ok_file


def _wrte_dataset_txt(dataset, save_path):
    data_set_txt = ''
    for i in dataset:
        data_set_txt += str(i[0]) + '┣┫' + str(i[1]) + '┣┫' + str(i[2]) + '\n'  # '\n'  # +
    f = open(save_path, 'w', encoding='utf-8')
    f.write(data_set_txt)
    f.close()


def cope_with_VOC():
    datalist = []
    voc_path = '/media/lg/DataSet_E/datasets/PASCAL_VOC/VOCdevkit'
    img_path = 'VOC2012/JPEGImages/'
    lab_path = 'VOC2012/Annotations/'
    txt_file = 'VOC2012/ImageSets/Main/trainval.txt'
    lines = open(os.path.join(voc_path,txt_file), 'r').readlines()
    for line in lines:
        line = line.strip()
        if os.path.isfile(os.path.join(voc_path, img_path, line + '.jpg')) and os.path.isfile(os.path.join(voc_path, lab_path, line + '.xml')):
            datalist.append([line + '.jpg', img_path + line + '.jpg', lab_path + line + '.xml'])
            print('adding:', datalist[-1])

    return datalist


if __name__ == '__main__':
    # pathes = ['D:/datasets/CCPD2019/ccpd_challenge/']
    # pathes = ['F:/datasets/SR/REDS4/train_sharp_part/']
    # pathes = ['F:/LG/OCR/PAN.pytorch-master/dadaset/wxf_ocr_data/']
    # pathes = ['E:/datasets/youku/youku_00200_00249_h_GT/']
    # pathes = ['E:/datasets/VisDrone2019/VisDrone2019-DET-train/']
    # img_path = ['E:/datasets/VisDrone2019/VisDrone2019-DET-train/images']
    # lab_path = 'E:/datasets/VisDrone2019/VisDrone2019-DET-train/annotations'
    # img_path = ['F:\Projects\\auto_Airplane\TS02\\20191220_1526019_20/']
    # lab_path = 'F:\Projects\\auto_Airplane\TS02\\20191220_1526019_20_refined/'

    # img_path = ['E:/datasets/VisDrone2019/VisDrone2019-VID-train/sequences/uav0000013_00000_v']
    # lab_path = 'E:\datasets\VisDrone2019\VisDrone2019-VID-train\\annotations/uav0000013_00000_v.txt'

    # img_path = 'VOC2012/JPEGImages'
    # lab_path = 'VOC2012/Annotations'
    # base_path = '/media/lg/DataSet_E/datasets/PASCAL_VOC/VOCdevkit'


    img_path = 'images'
    lab_path = 'labels'
    base_path = '/media/dell/data/ocr/电表识别/电表图片'

    save_path = 'util_tmp/make_list.txt'

    # datalist = make_list(base_path, img_path, lab_path)
    # datalist =_make_list_by_hand(path)


    img_path = '/media/dell/data/sleep/睡岗-分类-01_from_长寿-室内-分类-train'
    lab = ['是', '否']

    datalist = make_list_classfication(img_path, lab)
    # datalist = cope_with_VOC()

    _wrte_dataset_txt(datalist, save_path)


