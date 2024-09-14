'''
use the simplest way to track a car with template match.
writer: luogeng, 2020.03.10
'''

import cv2
import os
from lgdet.util.util_IQA import SameScore

object_car_path = 'E:/datasets/test_dataset/crop/car_track.png'
file = 'F:/datasets/SR/REDS4/train_sharp_part/000'

score = SameScore()

object_img = cv2.imread(object_car_path)
files = os.listdir(file)


# 输出视频路径
video_dir = 'video.avi'
# 帧率
fps = 24

# 图片尺寸
img_size = (1280, 720)

fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
video_writer = cv2.VideoWriter(video_dir, fourcc, fps, img_size)

for one_file in files:
    raw_img = cv2.imread(os.path.join(file, one_file))
    img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

    '''
    six ways of match template:
        ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    '''
    method = eval('cv2.TM_SQDIFF_NORMED')
    tmplate_img = cv2.cvtColor(object_img, cv2.COLOR_BGR2GRAY)
    final_score = 0.
    final_tl = None
    final_br = None
    for i in range(8):
        h, w = tmplate_img.shape[:2]
        tmp_return = cv2.matchTemplate(img, tmplate_img, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(tmp_return)

        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        img2 = raw_img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :]
        # cv2.imshow('img', img2)
        # cv2.waitKey()
        img1 = cv2.resize(object_img, (w, h))
        # cv2.imshow('img', img1)
        # cv2.waitKey()
        ssim_score = score.ssim(img1, img2)
        if ssim_score > final_score:
            final_score = ssim_score
            final_tl = top_left
            final_br = bottom_right
        tmplate_img = cv2.resize(tmplate_img, None, fx=0.9, fy=0.9)

    # 画矩形
    cv2.rectangle(raw_img, final_tl, final_br, 255, 2)
    cv2.imshow('img', raw_img)
    cv2.waitKey(1)

    video_writer.write(raw_img)

video_writer.release()


