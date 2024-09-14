import cv2
import numpy as np


def imgshow(img=None):
    cv2.imshow('str(img)', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


imgPath = 'E:/datasets/test_dataset/Image/'
_imgRaw = cv2.imread(imgPath + 'tileRefImageRGB.bmp')  # [:, :, 2]
_imgRaw = cv2.flip(_imgRaw, 0)
imgRaw = _imgRaw[600:3000, 2300:4500, ]
cv2.imwrite(imgPath + 'tileRefImageRGB.png', imgRaw)
# imgshow(imgRaw)
imgRaw_copy = imgRaw.copy()
img_gray = cv2.cvtColor(imgRaw_copy, cv2.COLOR_BGR2GRAY)
# imgshow(img_gray)
img_gray_aug = np.uint8(np.clip((1.5 * img_gray + 20), 0, 255))
# imgshow(cv2.resize(img_gray_aug, None, fx=0.4, fy=0.4))
ret, img_thresh_up = cv2.threshold(img_gray_aug[:2025], 130, 255, cv2.THRESH_BINARY)
ret_, img_thresh_down = cv2.threshold(img_gray_aug[2025:], 85, 255, cv2.THRESH_BINARY)
# imgshow(cv2.resize(img_thresh_down, None, fx=0.4, fy=0.4))
img_thresh = np.vstack((img_thresh_up, img_thresh_down))
# imgshow(cv2.resize(img_thresh, None, fx=0.4, fy=0.4))

img_mask = img_thresh.copy()
img_mask = cv2.bitwise_not(img_mask)  # 取反
# imgshow(mask_inv)

imgRaw_copy[img_mask == 0] = 255

# imgshow(imgRaw_copy)
cv2.imwrite(imgPath + 'tileRefImageRGBResult.png', imgRaw_copy)
