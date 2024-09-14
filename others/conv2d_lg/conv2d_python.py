import numpy as np


class Conv2DPython:
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.kernel = [np.random.random([in_channels, kernel_size, kernel_size]) for i in range(self.out_channels)]
        self.bias = [np.random.random([kernel_size, kernel_size]) for i in range(self.out_channels)]

    def forward(self, input):
        n, c, h, w = input.shape
        nout = []
        for ni in range(n):
            cki = []
            for ochi in range(self.out_channels):
                kwi = []
                for hi in range(0, h, self.stride):
                    khi = []
                    for wi in range(0, w, self.stride):
                        slice_input = input[ni, :, hi:hi + self.kernel_size, wi:wi + self.kernel_size]
                        if slice_input.shape != self.kernel[ochi].shape:
                            continue
                        wx_b = (slice_input * self.kernel[ochi] + self.bias[ochi]).sum()
                        khi.append(wx_b)
                    khi = np.asarray(khi)
                    if khi.size > 0:
                        kwi.append(khi)
                kwi = np.asarray(kwi)
                cki.append(kwi)
            cki = np.asarray(cki)
            nout.append(cki)
        nout = np.asarray(nout)
        return nout


if __name__ == '__main__':
    input = np.random.random([4, 3, 256, 256])
    conv = Conv2DPython(3, 32, 3, 2)
    y = conv.forward(input)
    print(y.shape)
