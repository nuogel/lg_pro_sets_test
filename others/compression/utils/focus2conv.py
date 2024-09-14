import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class CBL(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(CBL, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = CBL(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class FocusConv(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(FocusConv, self).__init__()
        self.conv = CBL(c1 * 4, c2, k, s, p, g, act)
        self.sliceconv = nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=1, stride=2, groups=c2, bias=False)
        self.sliceconv.weight.data.fill_(1)
        self.sliceconv.requires_grad = False

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        slice = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)

        x1 = x[:, :, 1:, :]
        x2 = x[:, :, :, 1:]
        x3 = x[:, :, 1:, 1:]

        y1 = self.sliceconv(x)
        y2 = self.sliceconv(x1)
        y3 = self.sliceconv(x2)
        y4 = self.sliceconv(x3)

        out = torch.cat([y1, y2, y3, y4], dim=1)

        assert slice.equal(out), 'slice.equal(out)->FALSE'
        return self.conv(out)


class FocusOther(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(FocusOther, self).__init__()
        self.conv_weight1 = nn.Parameter(torch.Tensor([0, 0, 0, 0, 1, 0, 0, 0, 0] * 3).view(3, 1, 3, 3))
        self.conv_weight1.requires_grad = False

        self.conv_weight2 = nn.Parameter(torch.Tensor([0, 0, 0, 0, 0, 1, 0, 0, 0] * 3).view(3, 1, 3, 3))
        self.conv_weight2.requires_grad = False

        self.conv_weight3 = nn.Parameter(torch.Tensor([0, 0, 0, 0, 0, 0, 0, 1, 0] * 3).view(3, 1, 3, 3))
        self.conv_weight3.requires_grad = False

        self.conv_weight4 = nn.Parameter(torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0, 1] * 3).view(3, 1, 3, 3))
        self.conv_weight4.requires_grad = False

        self.stride = 2
        self.pad = 1

    def forward(self, x: torch.Tensor):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        b, c, h, w = x.shape
        slice = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)

        out1 = F.conv2d(x, self.conv_weight1, None, stride=self.stride, padding=self.pad, groups=c)
        out2 = F.conv2d(x, self.conv_weight2, None, stride=self.stride, padding=self.pad, groups=c)
        out3 = F.conv2d(x, self.conv_weight3, None, stride=self.stride, padding=self.pad, groups=c)
        out4 = F.conv2d(x, self.conv_weight4, None, stride=self.stride, padding=self.pad, groups=c)
        out = torch.cat([out1, out3, out2, out4], 1)
        assert slice.equal(out), 'slice.equal(out)->FALSE'
        return torch.cat([out1, out3, out2, out4], 1)


class FocusConvLG(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(FocusConvLG, self).__init__()
        self.conv = CBL(c1 * 4, c2, k, s, p, g, act)
        self.sliceconv1 = nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=2, stride=2, groups=c2, bias=False)
        self.sliceconv1.weight.data.fill_(0)
        self.sliceconv1.weight.data[:, :, 0, 0] = 1
        self.sliceconv1.requires_grad = False

        self.sliceconv2 = nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=2, stride=2, groups=c2, bias=False)
        self.sliceconv2.weight.data.fill_(0)
        self.sliceconv2.weight.data[:, :, 1, 0] = 1
        self.sliceconv2.requires_grad = False

        self.sliceconv3 = nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=2, stride=2, groups=c2, bias=False)
        self.sliceconv3.weight.data.fill_(0)
        self.sliceconv3.weight.data[:, :, 0, 1] = 1
        self.sliceconv3.requires_grad = False

        self.sliceconv4 = nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=2, stride=2, groups=c2, bias=False)
        self.sliceconv4.weight.data.fill_(0)
        self.sliceconv4.weight.data[:, :, 1, 1] = 1
        self.sliceconv4.requires_grad = False

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        slice = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)

        y1 = self.sliceconv1(x)
        y2 = self.sliceconv2(x)
        y3 = self.sliceconv3(x)
        y4 = self.sliceconv4(x)

        out = torch.cat([y1, y2, y3, y4], dim=1)

        assert slice.equal(out), 'slice.equal(out)->FALSE'
        return self.conv(out)


if __name__ == '__main__':
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    x = torch.from_numpy(x)

    focus = FocusConv(3, 3)
    y = focus(x)

    f2 = FocusOther(3, 3)
    y = f2(x)

    focus = FocusConvLG(3, 3)
    y = focus(x)
