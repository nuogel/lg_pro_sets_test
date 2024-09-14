import cv2
import time
import numpy as np


def main(mingraythreh=20, minarea=30):
    cap = cv2.VideoCapture('/home/dell/ai_share/wending/高空抛物/2021-11-10高空抛物/ch01_00000000048000000.mp4')
    # 不设置是默认640*480，我们这里设置出来

    img_num = 0
    k = np.ones((3, 3), np.uint8)
    queue_tmies = []
    boxsequeece = []
    while True:
        success, img = cap.read()
        img = cv2.resize(img, None, fx=0.5, fy=0.5)
        localtime = time.asctime(time.localtime(time.time()))

        if not img_num:
            # 这里是由于第一帧图片没有前一帧
            previous = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_diff = cv2.absdiff(gray, previous)  # 计算绝对值差
        # previous 是上一帧图片的灰度图
        previous = gray
        thresh = cv2.threshold(gray_diff, mingraythreh, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.medianBlur(thresh, 3)

        close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        boxes = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < minarea:
                continue

            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            boxes.append([x, y, x + w, y + h])

        if boxes:
            queue_tmies.append(1)
        else:
            queue_tmies.append(0)

        queue_tmies = queue_tmies[:-30]
        boxsequeece.append(boxes)
        boxsequeece= boxsequeece[:-30]


        cv2.putText(img, localtime, (30, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
        # cv2.imshow("x", close)
        cv2.imshow("Result", img)
        img_num += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


main()
