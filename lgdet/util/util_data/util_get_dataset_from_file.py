import os
import glob
from sklearn.model_selection import train_test_split
from lgdet.util.util_get_cls_names import _get_class_names
import xml.etree.ElementTree as ET


def _get_train_test_dataset(cfg):
    x_dir = cfg.PATH.INPUT_PATH
    y_dir = cfg.PATH.LAB_PATH
    idx_stores_dir = 'DataLoader/datasets/tmp'
    test_train_ratio = cfg.TEST.TEST_SET_RATIO

    label_files = glob.glob('{}/*'.format(y_dir))
    x_files = glob.glob('{}/*'.format(x_dir))
    label_files_list = []
    x_extend_name = os.path.basename(x_files[0]).split('.')[-1]
    y_extend_name = os.path.basename(label_files[0]).split('.')[-1]
    if cfg.BELONGS == 'obd':
        class_map = _get_class_names(cfg.PATH.CLASSES_PATH)
        for label_file in label_files:
            file_name = os.path.basename(label_file).split('.')[0]
            img_file = os.path.join(cfg.PATH.INPUT_PATH, file_name + '.' + x_extend_name)
            if not os.path.isfile(img_file):
                continue
            if y_extend_name == 'txt':
                f = open(label_file, 'r')
                # print('checking file %s about:%s' % (label_file, cfg.TRAIN.CLASSES))
                for line in f.readlines():
                    cls = line.split(' ')[0]
                    if cls not in class_map:
                        continue
                    if class_map[cls] in cfg.TRAIN.CLASSES:
                        label_files_list.append([file_name, img_file, label_file])
                        break
            elif y_extend_name == 'xml':
                print(label_file)
                tree = ET.parse(label_file)
                root = tree.getroot()
                for object in root.findall('object'):
                    cls = object.find('name').text
                    if cls not in class_map:
                        continue
                    if class_map[cls] in cfg.TRAIN.CLASSES:
                        label_files_list.append([file_name, img_file, label_file])
                        break
    elif cfg.BELONGS == 'SR_DN':
        for label_file in label_files:
            file_name = os.path.basename(label_file).split('.')[0]
            x_file = os.path.join(cfg.PATH.INPUT_PATH, file_name + '.' + x_extend_name)
            if not os.path.isfile(x_file):
                continue
            label_files_list.append([file_name, x_file, label_file])
    elif cfg.BELONGS == 'ASR':
        x_files = glob.glob('{}/*.wav'.format(x_dir))
        for x_file in x_files:
            y_file = x_file + '.trn'
            if not os.path.isfile(y_file):
                continue
            file_name = os.path.basename(x_file).split('.')[0]
            label_files_list.append([file_name, x_file, y_file])

    else:
        label_files_list = label_files

    assert len(label_files_list) >= 1, 'No data found!'

    train_set, test_set = train_test_split(label_files_list, test_size=test_train_ratio)  # , random_state=1
    os.makedirs(idx_stores_dir, exist_ok=True)
    _wrte_dataset_txt((train_set, test_set), idx_stores_dir)
    # torch.save(train_set, os.path.join(idx_stores_dir, 'train_set'))
    # torch.save(test_set, os.path.join(idx_stores_dir, 'test_set'))
    print('saving train_set&test_set to %s' % idx_stores_dir)

    return train_set, test_set


def _wrte_dataset_txt(dataset, idx_stores_dir):
    train_set, test_set = dataset
    train_set_txt = ''
    test_set_txt = ''
    for i in train_set:
        train_set_txt += str(i[0]) + ';' + str(i[1]) + ';' + str(i[2]) + '\n'
    for i in test_set:
        test_set_txt += str(i[0]) + ';' + str(i[1]) + ';' + str(i[2]) + '\n'
    f = open(os.path.join(idx_stores_dir, 'train_set.txt'), 'w')
    f.write(train_set_txt)
    f.close()
    f = open(os.path.join(idx_stores_dir, 'test_set.txt'), 'w')
    f.write(test_set_txt)
    f.close()


def _read_train_test_dataset(cfg):
    def _load_from_file():
        train_set = []
        test_set = []
        for FILE_TXT in cfg.TRAIN.TRAIN_DATA_FROM_FILE:
            if FILE_TXT in ['CIFAR10']:
                from lgdet.util.util_data.cifar10 import load_cifar10
                train_set, test_set = load_cifar10()
            else:
                train_idx = os.path.join('datasets/', cfg.BELONGS, FILE_TXT,
                                         FILE_TXT + '_train' + '.txt')
                test_idx = os.path.join('datasets/', cfg.BELONGS, FILE_TXT,
                                        FILE_TXT + '_test' + '.txt')

                print('data set: %s' % FILE_TXT)
                f = open(train_idx, 'r', encoding='utf-8-sig')
                train_set.extend([line.strip().split('┣┫') for line in f.readlines()])
                f = open(test_idx, 'r', encoding='utf-8-sig')
                test_set.extend([line.strip().split('┣┫') for line in f.readlines()])
        return train_set, test_set

    if cfg.TEST.ONE_TEST:
        train_set = cfg.TEST.ONE_NAME
        if train_set == []:
            train_set, _ = _load_from_file()
            train_set = train_set[:1]
        test_set = train_set
    else:
        train_set, test_set = _load_from_file()
    return train_set, test_set


if __name__ == '__main__':
    y_dir = 'E://datasets//kitti//training//label_2//'
    idx_stores_dir = 'E://LG//programs//lg_pro_sets//tmp//'
    test_train_ratio = 0.1
    catgeorys = ['Car', ]
    _get_train_test_dataset(y_dir, idx_stores_dir, test_train_ratio)
