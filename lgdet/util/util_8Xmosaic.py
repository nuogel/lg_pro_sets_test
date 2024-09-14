import cv2
import numpy as np
import random

'''
NO USE. finally to be jpeg 压缩。
'''


def whsize_8x8(img, scale):
    # h, w = img.shape[:2]
    # nh = h // 2
    # nw = w // 2
    # aph = random.randint(0, nh - 8)
    # apw = random.randint(0, nw - 8)
    # img8x8 = img[nh - aph:nh - aph + 8, nw - apw:nw - apw + 8, :]

    # img8x8 = cv2.resize(img, None, fx=1/scale, fy=1/scale)
    a = random.random()
    if a < 0.5:
        np.random.shuffle(img[0:2])
    elif 0.5 < a < 0.8:
        v = np.var(img)
        if v < 100:
            img = cv2.medianBlur(img, ksize=5)
    else:
        pass
    img8x8 = img
    return np.clip(img8x8, 0, 255)


def mosaic_8x(img, scale):
    h, w = img.shape[:2]
    wh_size = 8
    w_blocks = w // wh_size
    h_blocks = h // wh_size
    one8x8_img = np.zeros((h_blocks * 8, w_blocks * 8, 3), dtype=np.uint8)
    for x in range(w_blocks):
        for y in range(h_blocks):
            # if (x + 2) * wh_size > w:
            #     x_end = w
            # else:
            #     x_end = (x + 1) * wh_size
            #
            # if (y + 2) * wh_size > h:
            #     y_end = h
            # else:
            #     y_end = (y + 1) * wh_size

            one_wsize_img = img[y * wh_size:(y + 1) * wh_size, x * wh_size:(x + 1) * wh_size, :]
            one8x8_img[y * 8:(y + 1) * 8, x * 8:(x + 1) * 8, :] = whsize_8x8(one_wsize_img, scale)
    img_mosaic = one8x8_img

    return img_mosaic


if __name__ == '__main__':
    imgpath = 'E:/LG/GitHub/lg_pro_sets/tmp/generated_labels/1.png'
    img = cv2.imread(imgpath)
    scale = 2
    img_mosaic = mosaic_8x(img, scale)
    cv2.imwrite('E:/LG/GitHub/lg_pro_sets/tmp/generated_labels/img_mosaic.png', img_mosaic)
    cv2.imshow('img', img_mosaic)
    cv2.waitKey()
