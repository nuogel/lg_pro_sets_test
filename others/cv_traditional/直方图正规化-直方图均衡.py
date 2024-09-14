import numpy as np
import cv2
import matplotlib.pyplot as plt
# 绘制图像灰度直方图

def deaw_gray_hist(gray_img):
    '''
    :param  gray_img大小为[h, w]灰度图像
    '''
    # 获取图像大小
    h, w = gray_img.shape
    gray_hist = np.zeros([256])
    for i in range(h):
        for j in range(w):
            gray_hist[gray_img[i][j]] += 1
    x = np.arange(256)
    # 绘制灰度直方图
    plt.bar(x, gray_hist)
    plt.xlabel("gray Label")
    plt.ylabel("number of pixels")
    plt.show()

# 读取图片
img_path='H:\LG\GitHub\lg_pro_sets\datasets\lena.jpg'
img = cv2.imread(img_path) # 这里需要指定一个 img_path
deaw_gray_hist(img[:,:,0])
cv2.imshow('ori_img', img)
cv2.waitKey()

# 对图像进行 线性变换
def linear_transform(img, a, b):
    '''
    我们把图像的灰度直方图看做是关于图像灰度值的一个函数，即每张图片都可以得到一个关于其灰度值的分布函数。我们可以通过线性变换让其灰度值的范围变大。
    假设图片上某点的像素值为 i，经过线性变换后得到的像素值为 o , a,b 为线性变换的参数则：
    o = a ∗ i + b
    其中当 a>0 时，图片的对比度会增大；当 0<a<1时，图片的对比度会减小。当 b>0 时，图片的亮度会增大；当 b<0时，图片的亮度会减小。

    :param img: [h, w, 3] 彩色图像
    :param a:  float  这里需要是浮点数，把图片uint8类型的数据强制转成float64
    :param b:  float
    :return: out = a * img + b

    '''
    out = a * img + b
    out[out > 255] = 255
    out = np.around(out)
    out = out.astype(np.uint8)
    return out

# # a = 2, b=10
# img = linear_transform(img, 2.0, 10)
# deaw_gray_hist(img[:, :, 0])
# cv2.imshow('linear_img', img)
# cv2.waitKey()


def normalize_transform(gray_img):
    '''
    直方图正规化
    :param gray_img:
    :return:
    '''
    Imin, Imax = cv2.minMaxLoc(gray_img)[:2]
    Omin, Omax = 0, 255
    # 计算a和b的值
    a = float(Omax - Omin) / (Imax - Imin)
    b = Omin - a * Imin
    out = a * gray_img + b
    out = out.astype(np.uint8)
    return out


b = img[:, :, 0]
g = img[:, :, 1]
r = img[:, :, 2]
b_out = normalize_transform(b)
g_out = normalize_transform(g)
r_out = normalize_transform(r)
nor_out = np.stack((b_out, g_out, r_out), axis=-1)
deaw_gray_hist(nor_out[:, :, 0])
cv2.imshow('nor_out', nor_out)
cv2.waitKey()
