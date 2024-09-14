from .siamrpn import TrackerSiamRPN
import cv2

'''
the performance on the test dataset is very good. will be useful !
'''


class LgTracker:
    def __init__(self):
        checkpoint = '../../../saved/checkpoint/SimRPN.pth'
        self.tracker = TrackerSiamRPN(checkpoint)

    def init_roi(self, img):
        ret = False
        roi = cv2.selectROI("tracking", img, False, False)  # x1, y1, w, h)
        if roi:
            self.tracker.init(img, roi)
            ret = True
        return ret

    def track_img(self, frame):
        x1, y1, w, h = self.tracker.update(frame)
        return int(x1), int(y1), int(x1 + w), int(y1 + h)
