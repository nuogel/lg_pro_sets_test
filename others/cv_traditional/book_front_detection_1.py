import cv2
import numpy as np


def imgshow(img=None):
    cv2.imshow(str(img), img)
    cv2.waitKey()
    cv2.destroyAllWindows()


imgPath = 'E:/datasets/test_dataset/Image/'
_imgRaw = cv2.imread(imgPath+'Image.bmp')  # [:, :, 2]
_imgRaw = cv2.flip(_imgRaw, 0)
imgRaw = _imgRaw[:,  2600:4600, ]
# cv2.imwrite(imgPath+'Image.png', imgRaw)
# imgshow(imgRaw)
imgRaw_copy = imgRaw.copy()
img_gray = cv2.cvtColor(imgRaw_copy, cv2.COLOR_BGR2GRAY)
# imgshow(img_gray)
img_gray_aug = np.uint8(np.clip((1.5 * img_gray + 10), 0, 255))
# imgshow(img_gray_aug)
# img_blur = cv2.blur(img_gray_aug, (3, 3))
# imgshow(img_blur)
ret, img_thresh = cv2.threshold(img_gray_aug, 55, 255, cv2.THRESH_BINARY)
# imgshow(img_thresh)
# img_canny = cv2.Canny(img_thresh, 8, 30)
# imgshow(img_canny)
# cv2.imwrite('E:/datasets/test_dataset/Image_canny.png', img_canny)

img_mask = img_thresh.copy()
x1, x2 = 520, 1449
mask_inv = cv2.bitwise_not(img_mask[:, x1:x2])  # 取反
# imgshow(mask_inv)
img_mask[:, x1:x2] = mask_inv
# imgshow(img_mask)

# _img_mask = cv2.bitwise_not(img_mask)
# imgshow(_img_mask)

# img_gray_aug = np.zeros_like(img_thresh) + 255
# imgshow(img_gray_aug)
imgRaw_copy[img_mask == 0] = 255

# img_dst = cv2.bitwise_and(imgRaw, imgRaw, mask=img_thresh)
# img_dst = cv2.add(imgRaw, _img_mask)

# imgRaw_copy = cv2.resize(imgRaw_copy, None, fx=0.5, fy=0.5)
# imgshow(imgRaw_copy)
cv2.imwrite(imgPath+'ImageResult.png', imgRaw_copy)

