import cv2
import numpy as np



def car_canny(img):
    # ---------原图
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # #---------二值化处理
    ret, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # cv2.threshold(img_gray, 90, 255, cv2.THRESH_BINARY)
    cv2.imshow('binary', binary)
    #
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))  # 定义结构元素
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)  # 开运算
    cv2.imshow('opening', binary)

    img_canny = cv2.Canny(binary, 80, 100)
    cv2.imshow('canny', img_canny)

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))  # 定义结构元素
    # closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # bi运算
    # cv2.imshow('closing', closing)

    # #---------锐化
    # # kernel = np.array([[-1, -1, -1], [-1, 11, -1], [-1, -1, -1]], np.float32)  # 高通滤波
    # kernel = np.array([[0, -1, 0], [-1, 7, -1], [0, -1, 0]], np.float32)  # 拉普拉斯算子(Laplacian)
    # img_sharp = cv2.filter2D(img, -1, kernel=kernel)
    # cv2.imshow('sharp', img_sharp)
    #
    # # #----------
    # img_pyr = cv2.bilateralFilter(opening, 50, 100, 100)  # a 邻域直径，b, c 分别是空间高斯函数标准差，灰度值相似性高斯函数标准差
    # img_pyr = cv2.pyrMeanShiftFiltering(img_pyr, 50, 100)
    #
    # cv2.imshow('img_pyr', img_pyr)
    # img_pry_gray = cv2.cvtColor(img_pyr, cv2.COLOR_BGR2GRAY)
    # ret, binary = cv2.threshold(img_pry_gray, 180, 255, cv2.THRESH_BINARY)
    # cv2.imshow('img_pry_gray', binary)
    image, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_out = []
    for i, cnt in enumerate(contours):
        hi = hierarchy[0][i]
        area = cv2.contourArea(cnt)  # 4386.5
        if hi[2] == -1 and hi[3] == -1 and area < 5000:
            continue
        contours_out.append(cnt)
        print(hi, area)
        cv2.drawContours(img, contours_out, -1, (0, 0, 255), 3)
        cv2.imshow('contors', img)


    fill_0 = cv2.fillPoly(img, pts=contours_out, color=(255, 255, 255))
    # cv2.imshow('fill_0', fill_0)

    color = (255, 255, 255)
    blue_img = np.zeros(img.shape).astype(img.dtype) + (255.0, 0, 0)
    cv2.fillPoly(blue_img,contours_out,color)
    cv2.imshow('blue_img', blue_img)

    # cv2.waitKey()
    # cv2.destroyAllWindows()