from socket import *
from time import ctime


class Server:
    def __init__(self):
        self.HOST = ''
        self.PORT = 21567
        self.BUFSIZ = 1024
        self.ADDR = (self.HOST, self.PORT)

        self.tcpServSock = socket(AF_INET, SOCK_STREAM)
        self.tcpServSock.bind(self.ADDR)
        self.tcpServSock.listen(5)

        print('waiting for connection...')
        self.tcpClientSock, addr = self.tcpServSock.accept()
        print('...connected from: ', addr)

    def start(self):
        while True:
            data = str(self.tcpClientSock.recv(self.BUFSIZ), encoding='utf-8')
            if not data:
                break
            self.tcpClientSock.send(bytes('[%s]server: I have received data from client: %s' % (ctime(), data,), 'utf-8'))
            print('received data from client:', data)

    def close(self):
        self.tcpClientSock.close()
        self.tcpServSock.close()


if __name__ == '__main__':
    server = Server()
    server.start()