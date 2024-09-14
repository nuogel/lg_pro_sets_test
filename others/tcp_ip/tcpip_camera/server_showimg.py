from socket import *

import cv2
import numpy as np


class Process:
    def __init__(self):
        self.init = False

    def init_process(self, data):
            self.init = True

    def run(self, data):
        cv2.imshow('URL2Image', data)
        c = cv2.waitKey(1) & 0xFF
        if c == 27 or c == ord('q'):
            self.init = False


class Server:
    def __init__(self):
        self.HOST = ''
        self.PORT = 21567
        self.BUFSIZ = 10240
        self.ADDR = (self.HOST, self.PORT)

        self.tcpServSock = socket(AF_INET, SOCK_STREAM)
        self.tcpServSock.bind(self.ADDR)
        self.tcpServSock.listen(5)
        self.counter = 0
        self.rest = None
        print('waiting for connection...')
        self.tcpClientSock, addr = self.tcpServSock.accept()
        print('...connected from: ', addr)
        self.tcpClientSock.send(bytes(str(self.ADDR), 'utf-8'))
        self.process = Process()
        self.start_frame = 0

    def start(self):
        while True:
            data = self.tcpClientSock.recv(self.BUFSIZ)
            self.counter += 1
            # data = bin(data)
            # print(self.counter, '-', len(data))
            if self.counter < self.start_frame: continue
            if self.rest:
                data = self.rest + data
            tmp = data.split(b'start') #b'start44484156'
            for img_byte in tmp:
                if img_byte[-3:] == b'end':
                    img_byte = img_byte[:-3]
                    image = np.asarray(bytearray(img_byte), dtype="uint8")
                    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                    if image is None:
                        print('img none')
                        continue
                    if self.counter >= self.start_frame and self.process.init:
                        loc = self.process.run(image)
                        self.tcpClientSock.send(bytes(str(loc), 'utf-8'))
                    else:
                        self.process.init_process(image)
                    self.rest = None
                else:
                    self.rest = img_byte

    def close(self):
        self.tcpClientSock.close()
        self.tcpServSock.close()


if __name__ == '__main__':
    server = Server()
    server.start()
