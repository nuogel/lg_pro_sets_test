# -*- coding: utf-8 -*-
"""A test file."""
import cv2
import torch
import numpy as np
from PIL import Image
import random

# import y4m


def _test_1():
    pic_path = 'E:\LG\programs\lg_pro_sets\datasets//1.jpg'
    # img = cv2.imread(pic_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.imshow('img', img)
    # cv2.waitKey()
    img = Image.open(pic_path)
    img = np.array(img)
    h = img.shape[0]
    w = img.shape[1]
    input = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).view(h, w, 3). \
        permute(2, 0, 1).float().div(255)
    end = 0


def _test_2():
    a = torch.randn(3)
    print(a)
    torch.sigmoid(a)
    print(a)
    print(np.exp(0.4058) * 4.0958478 * 16)
    print(np.exp(0.1243) * 9.108235 * 16)
    print((580 + 678.73) / 2, (150 + 314.92) / 2, 678.73 - 580, 314.92 - 150)

    print((39 + 0.3478) / 60 * 960)
    print((14 + 0.5287) / 24 * 384)

    x = torch.arange(5)
    print(x)
    mask = torch.gt(x, 1)  # 大于
    print(mask)
    print(x[mask])

    x = torch.arange(5)
    print(x)
    mask = torch.lt(x, 3)  # 小于
    print(mask)
    print(x[mask])

    x = torch.arange(5)
    print(x)
    mask = torch.eq(x, 3)  # 等于
    print(mask)
    print(x[mask])

    x = torch.Tensor([1, 2, 1, 0, 0])
    mask = torch.ne(x, 1)  # 非，一个数
    print(mask)
    print(x[mask])

    a = torch.Tensor([[0.6, 0.0, 0.0, 0.0], [0.0, 0.4, 0.0, 0.0],
                      [0.0, 0.0, 1.2, 0.0], [0.0, 0.0, 0.0, -0.4]])
    mask = torch.nonzero(a)  # 非零
    print(mask)
    print(torch.numel(mask))
    print(torch.numel(a))
    # print(a[mask])
    print(torch.numel(mask) / torch.numel(a))

    print(pow(0.995, 100))


def _y4m():
    f = 'E:/datasets/SR/youku/youku_00000_00049_l/Youku_00000_l.y4m'
    img = y4m.Reader(f)
    a = 0


def other():
    n = 'GroundDisconnector_0\
         Capacitor_90\
         Capacitor_0\
         Disconnector_90\
         CBreaker_90\
         Transformer2_180\
         Transformer3_12_0\
         CBreaker_0\
         GroundDisconnector_180\
         ACLineEnd_270\
         text\
         ACLineEnd_90\
         ACLineEnd_0\
         Reactor_0\
         Transformer3_12_270\
         GroundDisconnector_270\
         Capacitor_270\
         SVG_0\
         Transformer3_34_90\
         Reactor_270\
         Disconnector_180\
         Transformer3_34_0\
         Bus_0\
         Disconnector_0\
         other\
         ACLineEnd_180\
         Disconnector_270\
         Transformer2_270\
         Bus_90\
         Transformer2_0\
         GroundDisconnector_90\
         Transformer2_90\
         Generator_0\
         Bridge'

    n = n.split('         ')
    n.sort()
    for i in n:
        print(i)


def resize():
    img = cv2.imread('/media/dell/data/比赛/提交/logo.png')
    img = cv2.resize(img, None, fx=10, fy=10)
    cv2.imwrite('/media/dell/data/比赛/提交/logo2.png', img)



if __name__ == '__main__':
    # _test_1()
    # a = torch.version.cuda
    # _y4m()
    # other()
    # resize()
    random.randint(0,1)
    a= torch.full(size=(16,), fill_value=26, dtype=torch.long)
    a=0