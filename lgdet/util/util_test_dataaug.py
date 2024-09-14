"""Test the functions or data that you want."""

import os
import time
from util import Dataaug
from lgdet.util.util_yml_parse import parse_yaml


def test_time_data_aug():
    """Compare the time of a list and one image."""
    print('testing data augmentation...')
    idx = []
    for i in range(0, 32):
        idx = idx + [i]
    time_a = time.time()
    # augmentation(idx)
    time_b = time.time()
    for i in range(0, 32):
        augmentation([i], do_aug=False)
    time_c = time.time()
    print('image augmentation for a list with 32 images: ',
          time_b - time_a,
          '\nimage augmentation for one image per time for 32 times: ',
          time_c - time_b)


def test_data_aug_and_show():
    real_path = os.path.realpath('../../../')
    """Show the images with data augmentation."""
    path = '../../datasets/kitti/training/image_2/'
    path = '../../datasets/goldenridge_testset/test_images_in_lane_full/imgs/'
    cfg = parse_yaml(os.path.join(real_path,'cfg/fcos.yml'), real_path)

    data_aug = Dataaug(cfg)

    files = os.listdir(path)
    files.sort()
    for i, file in enumerate(files):
        print(file)
        file_name = os.path.basename(file).split('.')[0]
        data_aug._show_imgaug(['20190412_0132'], do_aug=False, resize=False, show_img=True)


if __name__ == "__main__":
    test_data_aug_and_show()
    # test_time_data_aug()
