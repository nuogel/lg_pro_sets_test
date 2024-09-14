import argparse
import cv2
from kcf import Tracker


def main():
    save_reslut = 1
    vid_path = 'car.avi'
    cap = cv2.VideoCapture(vid_path)
    tracker = Tracker()
    ok, frame = cap.read()
    if not ok:
        print("error reading video")
        exit(-1)
    roi = cv2.selectROI("tracking", frame, False, False)
    # roi = (218, 302, 148, 108)
    tracker.init(frame, roi)
    i = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        (x, y, w, h), response = tracker.update(frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
        cv2.imshow('tracking', frame)
        c = cv2.waitKey(1) & 0xFF
        if c == 27 or c == ord('q'):
            break

        if save_reslut:
            imgpath = 'demo_result/images/%05d.jpg' % i
            labpath = 'demo_result/labels/%05d.txt' % i
            cv2.imwrite(imgpath, frame)
            f = open(labpath, 'w')
            txt = ','.join(list(map(str, [x, y, w, h]))) + '\n'
            f.write(txt)
            f.close()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
