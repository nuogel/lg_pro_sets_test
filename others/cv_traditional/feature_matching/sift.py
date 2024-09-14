import cv2

def sift(filename):
    img = cv2.imread(filename)  # 读取文件
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转化为灰度图
    sift = cv2.xfeatures2d.SIFT_create()
    keyPoint, descriptor = sift.detectAndCompute(img, None)  # 特征提取得到关键点以及对应的描述符（特征向量）
    return img, keyPoint, descriptor
