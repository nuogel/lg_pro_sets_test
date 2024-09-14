"""Convert goldenridge data to kitti shape."""
import os
from cfg import config

IMG_PATH = config.IMGPATH
LAB_PATH = config.LABPATH


def _change_file_name():
    i = 0
    for file in os.listdir(IMG_PATH):
        one_labpath = os.path.join(LAB_PATH, file.split('.')[0] + '.txt')
        if os.path.isfile(one_labpath):
            os.rename(os.path.join(IMG_PATH, file), os.path.join(IMG_PATH, '%06d.jpg' % i))
            os.rename(one_labpath, os.path.join(LAB_PATH, '%06d.txt' % i))
            i = i + 1


if __name__ == "__main__":
    print('be care of renaming')
    # _change_file_name()
