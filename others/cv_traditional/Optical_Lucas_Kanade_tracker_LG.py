# encoding:utf-8
'''
Lucas-Kanade tracker
====================
Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.
Usage
-----
lk_track.py [<video_source>]
Keys
----
ESC - exit
'''

import numpy as np
import cv2
# from common import anorm2, draw_str
from time import clock

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=500,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)


class App:
    def __init__(self, video_src):  # 构造方法，初始化一些参数和视频路径
        self.track_len = 100
        self.detect_interval = 5
        self.tracks = []
        self.cam = cv2.VideoCapture(video_src)
        self.frame_idx = 0
        self.distance = 1

    def run(self):  # 光流运行方法
        while True:
            ret, frame = self.cam.read()  # 读取视频帧
            if ret == True:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转化为灰度虚图像
                vis = frame.copy()

                if len(self.tracks) > 0:  # 检测到角点后进行光流跟踪
                    img0, img1 = self.prev_gray, frame_gray
                    p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                    p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)  # 前一帧的角点和当前帧的图像作为输入来得到角点在当前帧的位置
                    p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)  # 当前帧跟踪到的角点及图像和前一帧的图像作为输入来找到前一帧的角点位置
                    d = abs(p0 - p0r).reshape(-1, 2).max(-1)  # 得到角点回溯与前一帧实际角点的位置变化关系
                    good = d < self.distance  # 判断d内的值是否小于distance，大于distance跟踪被认为是错误的跟踪点
                    new_tracks = []
                    for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):  # 将跟踪正确的点列入成功跟踪点
                        if not good_flag:
                            continue
                        tr.append((x, y))
                        if len(tr) > self.track_len:
                            del tr[0]
                        new_tracks.append(tr)
                        cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                    self.tracks = new_tracks
                    cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))  # 以上一振角点为初始点，当前帧跟踪到的点为终点划线
                    # draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))
                else:
                    lt = (519, 392.0)
                    for i in range(15):
                        for j in range(15):
                            self.tracks.append([(lt[0] + i * 5, lt[1] + j*5)])
                            self.tracks.append([(lt[0] + j*5, lt[1] + i * 5)])
                # if self.frame_idx % self.detect_interval == 0:  # 每5帧检测一次特征点
                #     mask = np.zeros_like(frame_gray)  # 初始化和视频大小相同的图像
                #     mask[:] = 255  # 将mask赋值255也就是算全部图像的角点
                #     for x, y in [np.int32(tr[-1]) for tr in self.tracks]:  # 跟踪的角点画圆
                #         cv2.circle(mask, (x, y), 5, 0, -1)
                #     p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)  # 像素级别角点检测
                #     if p is not None:
                #         for x, y in np.float32(p).reshape(-1, 2):
                #             self.tracks.append([(x, y)])  # 将检测到的角点放在待跟踪序列中

                self.frame_idx += 1
                self.prev_gray = frame_gray
                cv2.imshow('lk_track', vis)
                cv2.waitKey()

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break


def main():
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = "E:/datasets/SR_video_test/blur-64/raw_video/000_x5.mp4"

    print(__doc__)
    App(video_src).run()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
