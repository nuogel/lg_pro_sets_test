import cv2
import numpy as np

'''
Failed...
'''

video_src = "E:/datasets/SR_video_test/blur-64/raw_video/000_x5.mp4"
cap = cv2.VideoCapture(video_src)

# ret判断是否读到图片
# frame读取到的当前帧的矩阵
# 返回的是元组类型，所以也可以加括号
ret, frame = cap.read()
# print(type(ret), ret)
# print(type(frame), frame)


# 设置跟踪框参数
x, y, w, h = 519, 392, 50, 50  # simply hardcoded the values
track_window = (x, y, w, h)

# 从当前帧中框出一个小框
roi = frame[y:y + h, x:x + w]

# RGB转为HSV更好处理
hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# inRange函数设置亮度阈值
# 去除低亮度的像素点的影响
# eg. mask = cv2.inRange(hsv, lower_red, upper_red)

# 将低于和高于阈值的值设为0
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

# 然后得到框中图像的直方图
# cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate ]])
# mask 即上文的阈值设置
# histSize表示这个直方图分成多少份（即多少个直方柱）
# range是表示直方图能表示像素值的范围
# 返回直方图
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])

# 归一化函数cv2.normalize(src[, dst[, alpha[, beta[, norm_type[, dtype[, mask]]]]]])
# 返回dst类型
# 归一化就是要把需要处理的数据经过处理后（通过某种算法）限制在你需要的一定范围内
# src  - 输入数组
# dst  - 与src大小相同的输出数组
# alpha  - 范围值，   以便在范围归一化的情况下归一化到较低范围边界
# beta  - 范围归一化时的上限范围; 它不用于标准规范化
# normType  - 规范化类型 这里的NORM_MINMAX是数组的数值被平移或缩放到一个指定的范围，线性归一化。
# dtype  - 当为负数时，输出数组与src的类型相同；否则，它具有与src相同的通道数；深度=CV_MAT_DEPTH（dtype）
# mask  - 可选的操作掩码。
# cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# 设置迭代的终止标准，最多十次迭代
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while (1):
    ret, frame = cap.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 反向投影函数（特征提取函数）
        # 反向投影是一种记录给定图像中的像素点如何适应直方图模型像素分布的方式
        # 反向投影就是首先计算某一特征的直方图模型，然后使用模型去寻找图像中存在的特征
        # cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])

        # images:待处理的图像，图像格式为uint8或float32
        # channels:对应图像需要统计的通道，若是灰度图则为0，彩色图像B、G、R对应0、1、2
        # mask:掩膜图像。如果统计整幅图像就设置为None，否则这里传入设计的掩膜图像。
        # histSize表示这个直方图分成多少份（即多少个直方柱）
        # ranges:像素量化范围，通常为0 - 255。
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # RotatedRect CamShift(InputArray probImage, Rect&window, TermCriteria criteria)。
        # probImage为输入图像直方图的反向投影图，
        # window为要跟踪目标的初始位置矩形框，
        # criteria为算法结束条件。
        # 函数返回一个有方向角度的矩阵。
        #
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)

        # Draw it on image
        pts = cv2.boxPoints(ret)

        # 类型转换int0()用于索引的整数(same as C ssize_t; normally either int32 or int64)
        pts = np.int0(pts)

        # 非填充多边形：polylines()
        # cv2.polylines(img, pts, isClosed, color[, thickness[, lineType[, shift]]])
        # img – 要画的图片
        # pts – 多边形的顶点
        # isClosed – 是否闭合线段
        # color – 颜色
        img2 = cv2.polylines(frame, [pts], True, 255, 2)

        cv2.imshow('img2', img2)

        # 停止追踪按钮
        k = cv2.waitKey(1000) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k) + ".jpg", img2)

    else:
        break

cv2.destroyAllWindows()
cap.release()
