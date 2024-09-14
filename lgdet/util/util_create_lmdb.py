# coding=utf-8
# -*- coding: UTF-8 -*-
import os
import lmdb
import cv2
import numpy as np
from tqdm import tqdm
import shutil


class Imdb_maker:
    def __init__(self):
        pass

    def creater(self, outputPath, train_set):
        """
        Create LMDB dataset for training.
        """
        train_set=train_set[:10]
        nSamples = len(train_set)
        data_size_per_img = cv2.imread(train_set[0][1], cv2.IMREAD_UNCHANGED).nbytes
        print('data size per image is: ', data_size_per_img)
        data_size = data_size_per_img * nSamples
        env = lmdb.open(outputPath, map_size=data_size)
        cache = {}
        cnt = 1
        for i in range(nSamples):
            imagePath = train_set[i][1]
            labelPath = train_set[i][2]
            # 数据库中都是二进制数据
            image = open(imagePath, 'rb').read()
            label = open(labelPath, 'rb').read()

            cache[imagePath] = image
            cache[labelPath] = label

            if cnt % 1000 == 0:
                self.writeCache(env, cache)
                cache = {}
                print('Written %d / %d' % (cnt, nSamples))
            cnt += 1
        nSamples = cnt - 1
        cache['num-samples'] = str(nSamples).encode()
        self.writeCache(env, cache)
        print('Created dataset with %d samples' % nSamples)

    def reader(self, imdb_path, idx):
        self.env = lmdb.open(imdb_path)
        self.txn = self.env.begin()

        img_code = self.txn.get(idx[1].encode())
        img = np.frombuffer(img_code, np.uint8)  # 转成8位无符号整型
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)  # 解码
        # cv2.imshow('img',img)
        # cv2.waitKey()
        label = self.txn.get(idx[2].encode())
        return img, label

    def writeCache(self, env, cache):
        with env.begin(write=True) as txn:
            for k, v in cache.items():
                txn.put(k.encode(), v)


if __name__ == "__main__":
    def _read_train_test_dataset(idx_stores_dir):
        print('reading train_set&test_set from %s' % idx_stores_dir)
        f = open(os.path.join(idx_stores_dir, 'train_set.txt'), 'r')
        train_set = [line.strip().split(';') for line in f.readlines()]
        f = open(os.path.join(idx_stores_dir, 'test_set.txt'), 'r')
        test_set = [line.strip().split(';') for line in f.readlines()]
        return train_set, test_set


    txt_path = 'E:/LG/GitHub/lg_pro_sets/dataloader/OBD_idx_stores/used/windows/w_kitti/'
    lmdb_output_path = 'E:/datasets/kitti/lmdb/'
    if os.path.isdir(lmdb_output_path):
        shutil.rmtree(lmdb_output_path)
    imdb = Imdb_maker()
    train_set, test_set = _read_train_test_dataset(txt_path)
    imdb.creater(lmdb_output_path, train_set)
    imdb.reader(lmdb_output_path, train_set[0])
