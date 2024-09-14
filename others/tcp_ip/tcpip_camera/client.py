from socket import *
import cv2
import numpy as np
import time


class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

    def get_one_frame(self):
        # print('reading')
        ret, frame = self.cap.read()
        if ret == False:
            print('reading error')
        return ret, frame


class Client:
    def __init__(self):
        self.HOST = '192.168.0.104'
        self.PORT = 21567
        self.BUFSIZ = 1024
        self.ADDR = (self.HOST, self.PORT)
        self.tcpClientSock = socket(AF_INET, SOCK_STREAM)
        self.tcpClientSock.connect(self.ADDR)
        self.camera = Camera()
        self.counter = 0

    def start_client(self):

        while True:
            data = self.camera.get_one_frame()
            # data = cv2.imread('/media/lg/SSD_WorkSpace/LG/GitHub/lg_pro_sets/saved/denoise/edsr_1216x554x2.png')
            time.sleep(0.1)
            ret, fram = data
            # img_encode = np.reshape(data, (-1, 1))
            _, img_encode = cv2.imencode(".jpg", fram)
            data_encode = np.array(img_encode)
            img_bytes = data_encode.tostring()
            # length = len(img_bytes)
            send_byte = bytes('start', encoding='utf-8') + img_bytes + bytes('end', encoding='utf-8')
            # img_bytes = bytes(img_bytes, 'utf-8')
            if not ret:
                continue
            self.counter += 1

            feedback = self.tcpClientSock.send(send_byte)
            print(self.counter, '-', len(send_byte))
            if feedback != len(send_byte):
                print(feedback)

            data = self.tcpClientSock.recv(self.BUFSIZ)
            if not data:
                break
            print(str(data, encoding='utf-8'))

    def close_clent(self):
        self.tcpClientSock.close()
        print('closing')


if __name__ == '__main__':
    camera = Client()
    try:
        camera.start_client()
    except:
        camera.close_clent()
