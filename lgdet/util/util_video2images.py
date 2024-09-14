import cv2
import os


def video2images(file_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    cap = cv2.VideoCapture(file_path)
    rec = True
    i = 0
    while rec:
        rec, img = cap.read()
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        if rec:
            if i % 1 == 0:
                save_path = os.path.join(output_path, '%06d.png' % i)
                cv2.imwrite(save_path, img)
                print('saving: ', save_path)
            i += 1


if __name__ == '__main__':
    file_path = 'E:/for_test/fly3.mp4'
    output_path = 'E:/for_test/fly3'
    video2images(file_path, output_path)
