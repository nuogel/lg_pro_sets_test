import cv2
import os

'''
VISDRONE:
 VID-dataset:
    sequence name.txt ->{index name of .jpg, target_index,  x, y, w, h, score, class, x,x,x }
                        1,0,593,43,174,190,0,0,0,0
                        2,0,592,43,174,189,0,0,0,0
                        3,0,592,43,174,189,0,0,0,0
                        4,0,592,43,174,189,0,0,0,0
                        5,0,592,43,174,189,0,0,0,0
'''


def make_VisDrone2019_VID_dataset(path, show_images=False):
    lables_base = os.path.join(path, 'annotations')
    imgpath_base = os.path.join(path, 'sequences')
    name_dict = {'0': 'ignored regions', '1': 'pedestrian', '2': 'people',
                 '3': 'bicycle', '4': 'car', '5': 'van', '6': 'truck',
                 '7': 'tricycle', '8': 'awning-tricycle', '9': 'bus',
                 '10': 'motor', '11': 'others'}
    name_dict_tansform = {'0': 0, '1': 2, '2': 2,
                          '3': 3, '4': 1, '5': 1, '6': 1,
                          '7': 3, '8': 3, '9': 1,
                          '10': 3, '11': 0}
    all_info_list = []
    for file in os.listdir(lables_base):
        info_dict = {}
        for line in open(os.path.join(lables_base, file)).readlines():
            tmps = line.split(',')
            # if name == 'ignored regions':
            #     continue
            img_id = int(tmps[0])
            _img_id = '%07d.jpg' % (img_id)
            img_file = os.path.basename(file).split('.')[0]
            img_path = os.path.join(imgpath_base, img_file, _img_id)
            name = name_dict_tansform[tmps[7]]

            xmin = int(tmps[2])
            ymin = int(tmps[3])
            xmax = xmin + int(tmps[4])
            ymax = ymin + int(tmps[5])

            if img_id in info_dict.keys():
                info_dict[img_id].append([name, xmin, ymin, xmax, ymax])
            else:
                info_dict[img_id] = [img_path, [name, xmin, ymin, xmax, ymax]]
        all_info_list.append(info_dict)
        break
    if show_images:
        for info_dict in all_info_list:
            for k, v in info_dict.items():
                img_path = v[0]
                images = cv2.imread(img_path)
                for vi in v[1:]:
                    [name, xmin, ymin, xmax, ymax] = vi
                    cv2.rectangle(images, (xmin, ymin), (xmax, ymax), (255, 0, 0))
                    cv2.putText(images, str(name_dict[str(name)]), (xmin, ymin), 1, 1, (0, 0, 255))
                cv2.imshow('img', images)
                cv2.waitKey()
    return all_info_list


if __name__ == '__main__':
    make_VisDrone2019_VID_dataset(path='E:/datasets/VisDrone2019/VisDrone2019-VID-train/', show_images=True)
