import cv2
import numpy as np


def _viz_flow(inputs=None, flow=None, predicted=None, show_time=10000):
    # 色调H：用角度度量，取值范围为0°～360°，从红色开始按逆时针方向计算，红色为0°，绿色为120°,蓝色为240°
    # 饱和度S：取值范围为0.0～1.0
    # 亮度V：取值范围为0.0(黑色)～1.0(白色)
    inputs_1 = None
    inputs_2 = None
    bgr1 = None
    bgr2 = None
    show_list = []

    def flo2BGR(flow):
        flow = np.asarray(flow.detach().cpu())
        h, w = flow.shape[:2]
        hsv = np.zeros((h, w, 3), np.uint8)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # flownet是将V赋值为255, 此函数遵循flownet，饱和度S代表像素位移的大小，亮度都为最大，便于观看
        # 也有的光流可视化讲s赋值为255，亮度代表像素位移的大小，整个图片会很暗，很少这样用
        hsv[..., 2] = 255
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr

    if inputs:
        inputs_1 = np.asarray(inputs[0], dtype=np.uint8)
        inputs_2 = np.asarray(inputs[1], dtype=np.uint8)
        show_list.append(inputs_1)
        show_list.append(inputs_2)
    if flow is not None:
        bgr1 = flo2BGR(flow)
        show_list.append(bgr1)

    if predicted is not None:
        bgr2 = flo2BGR(predicted)
        show_list.append(bgr2)

    img = np.concatenate(show_list, 1)
    H = img.shape[0]
    if H < 100:
        img = cv2.resize(img, None, fx=4, fy=4)
    cv2.imshow('img', img)
    cv2.waitKey(show_time)
