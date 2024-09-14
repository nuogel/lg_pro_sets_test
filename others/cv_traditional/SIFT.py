import numpy as np
import cv2

imgpath = 'E:/datasets/kaggle_DOG_vs_CAT/train/dog.1.jpg'
img = cv2.imread(imgpath)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
# 找出关键点
kp = sift.detect(gray, None)
# 计算关键点特征向量
kp, vector = sift.compute(gray, kp)
# 对关键点进行绘图
ret = cv2.drawKeypoints(gray, kp, img)
cv2.imshow('ret', ret)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 使用关键点找出sift特征向量
kp, des = sift.compute(gray, kp)

print(np.shape(kp))
print(np.shape(des))

print(des[0])