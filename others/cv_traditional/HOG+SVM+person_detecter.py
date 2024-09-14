# import the necessary packages
from imutils.object_detection import non_max_suppression
import numpy as np
import imutils
import cv2
import glob

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# cap = cv2.VideoCapture('img/test.mp4')
files = glob.glob('E:/datasets/TRACK/LG/images/*.jpg')

# load the image and resize it to (1) reduce detection time
# and (2) improve detection accuracy
for file in files:
    image = cv2.imread(file)
    # image = cv2.imread('img/test5.jpg')
    image = imutils.resize(image, width=min(400, image.shape[1]))
    orig = image.copy()

    # detect people in the image
    (rects, weights) = hog.detectMultiScale(
        image, winStride=(4, 4), padding=(8, 8), scale=1.05
    )

    # draw the original bounding boxes
    # for (x, y, w, h) in rects:
    #     cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

    #     # apply non-maxima suppression to the bounding boxes using a
    #     # fairly large overlap threshold to try to maintain overlapping
    #     # boxes that are still people
    #     rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])

    pick = non_max_suppression(rects, probs=1, overlapThresh=0.15)
    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

    # show the output images
    # cv2.imshow("Before NMS", orig)
    cv2.imshow("After NMS", image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
