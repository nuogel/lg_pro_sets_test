# 均值模糊、中值模糊、自定义模糊    模糊是卷积的一种表象
import cv2
import numpy as np


def blur_demo(image):  # 均值模糊  去随机噪声有很好的去燥效果
    dst = cv2.blur(image, (1, 15))  # （1, 15）是垂直方向模糊，（15， 1）还水平方向模糊
    # cv2.namedWindow('blur_demo', cv2.WINDOW_NORMAL)
    cv2.imshow("blur_demo", dst)
    return dst


def median_blur_demo(image):  # 中值模糊  对椒盐噪声有很好的去燥效果
    dst = cv2.medianBlur(image, 5)
    # cv2.namedWindow('median_blur_demo', cv2.WINDOW_NORMAL)
    cv2.imshow("median_blur_demo", dst)
    return dst


def custom_blur_demo(image):  # 用户自定义模糊
    kernel = np.ones([5, 5], np.float32) / 25  # 除以25是防止数值溢出
    dst = cv2.filter2D(image, -1, kernel)
    # cv2.namedWindow('custom_blur_demo', cv2.WINDOW_NORMAL)
    cv2.imshow("custom_blur_demo", dst)
    return dst


# 边缘保留滤波（EPF）  高斯双边、均值迁移
def bi_demo(image):  # 双边滤波
    dst = cv2.bilateralFilter(image, 0, 100, 15) #9 邻域直径，两个 75 分别是空间高斯函数标准差，灰度值相似性高斯函数标准差
    # cv2.namedWindow("bi_demo", cv2.WINDOW_NORMAL)
    cv2.imshow("bi_demo", dst)
    return dst


def shift_demo(image):  # 均值迁移
    dst = cv2.pyrMeanShiftFiltering(image, 10, 50)
    # cv2.namedWindow("shift_demo", cv2.WINDOW_NORMAL)
    cv2.imshow("shift_demo", dst)
    return dst


imgpath = 'E:/LG/GitHub/lg_pro_sets/tmp/generated_labels/rdn_predict.png'

src = cv2.imread(imgpath)
cv2.imshow('input_image', src)

blur_demo(src)
median_blur_demo(src)
custom_blur_demo(src)
bi_demo(src)
shift_demo(src)

cv2.waitKey(0)
cv2.destroyAllWindows()
