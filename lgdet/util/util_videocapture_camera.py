import cv2


class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

    def get_one_frame(self):
        print('reading')
        ret, frame = self.cap.read()
        if ret == False:
            print('reading error')
        return ret, frame


if __name__ == '__main__':
    camera = Camera()
    ret = True
    while ret:
        ret, frame = camera.get_one_frame()
        cv2.imshow('img', frame)
        cv2.waitKey(1)
