from socket import *


class Client:
    def __init__(self):
        self.HOST = 'localhost'
        self.PORT = 21567
        self.BUFSIZ = 1024
        self.ADDR = (self.HOST, self.PORT)
        self.tcpClientSock = socket(AF_INET, SOCK_STREAM)
        self.tcpClientSock.connect(self.ADDR)

    def start_client(self):

        while True:
            data = input('> ')
            if not data:
                break
            self.tcpClientSock.send(bytes(data, 'utf-8'))
            data = self.tcpClientSock.recv(self.BUFSIZ)
            if not data:
                break
            print(str(data, encoding='utf-8'))

    def close_clent(self):
        self.tcpClientSock.close()


if __name__ == '__main__':
    client = Client()
    client.start_client()