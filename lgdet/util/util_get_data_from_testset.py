import os.path as path
import shutil
import torch
from cfg import config

testdata_idx = torch.load(path.join(config.TMP_DIR, 'idx_stores/test_set'))
print(testdata_idx)
for idx in testdata_idx:
    test_images_path = path.join(config.IMGPATH, '%06d.jpg' % idx)
    test_labels_path = path.join(config.LABPATH, '%06d.txt' % idx)
    shutil.copy(test_images_path, './for_tgj/images')
    shutil.copy(test_labels_path, './for_tgj/labels')
