import torch.nn as nn
import torch
from .activation_funs import *


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DeConv(nn.Module):
    def __init__(self, c1, c2, k=2, s=2, bias=False):  # ch_in, ch_out, kernel, stride, padding, groups
        super(DeConv, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=c1, out_channels=c2, kernel_size=k, stride=s, groups=c2, padding=0, output_padding=0, bias=bias)
        self.init_params()

    def init_params(self):
        shape = self.deconv.weight.data.shape
        self.deconv.weight.data = torch.ones(shape)
        self.deconv.weight.requires_grad = False

    def forward(self, x):
        # assert self.deconv.weight.requires_grad == False
        return self.deconv(x)


class CBL(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(CBL, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish() if act else nn.Identity()
        # self.act = nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()

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


class FocusConvLG(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(FocusConvLG, self).__init__()
        self.conv = CBL(c1 * 4, c2, k, s, p, g, act)
        self.sliceconv1 = nn.Conv2d(in_channels=c1, out_channels=c1, kernel_size=2, stride=2, groups=c1, bias=False)
        self.sliceconv1.weight.data.fill_(0)
        self.sliceconv1.weight.data[:, :, 0, 0] = 1
        self.sliceconv1.weight.requires_grad = False

        self.sliceconv2 = nn.Conv2d(in_channels=c1, out_channels=c1, kernel_size=2, stride=2, groups=c1, bias=False)
        self.sliceconv2.weight.data.fill_(0)
        self.sliceconv2.weight.data[:, :, 1, 0] = 1
        self.sliceconv2.weight.requires_grad = False

        self.sliceconv3 = nn.Conv2d(in_channels=c1, out_channels=c1, kernel_size=2, stride=2, groups=c1, bias=False)
        self.sliceconv3.weight.data.fill_(0)
        self.sliceconv3.weight.data[:, :, 0, 1] = 1
        self.sliceconv3.weight.requires_grad = False

        self.sliceconv4 = nn.Conv2d(in_channels=c1, out_channels=c1, kernel_size=2, stride=2, groups=c1, bias=False)
        self.sliceconv4.weight.data.fill_(0)
        self.sliceconv4.weight.data[:, :, 1, 1] = 1
        self.sliceconv4.weight.requires_grad = False

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        # slice = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        # print('self.sliceconv1.requires_grad: ', self.sliceconv1.weight.requires_grad)
        # print('self.sliceconv2.requires_grad: ', self.sliceconv2.weight.requires_grad)
        # print('self.sliceconv3.requires_grad: ', self.sliceconv3.weight.requires_grad)
        # print('self.sliceconv4.requires_grad: ', self.sliceconv4.weight.requires_grad)
        # print('self.sliceconv4.weight.data', self.sliceconv4.weight.data)
        y1 = self.sliceconv1(x)
        y2 = self.sliceconv2(x)
        y3 = self.sliceconv3(x)
        y4 = self.sliceconv4(x)

        out = torch.cat([y1, y2, y3, y4], dim=1)

        # assert slice.equal(out), 'slice.equal(out)->FALSE'
        return self.conv(out)


'''
An alternative implementation for PyTorch with auto-infering the x-y dimensions.
'''


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = CBL(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = CBL(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = CBL(c1, c_, 1, 1)
        self.cv2 = CBL(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = CBL(c1, c_, 1, 1)
        self.cv2 = CBL(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class AddCoords(nn.Module):

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class Conv2D_Norm_Activation(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='mish', norm_type='BN', num_groups=32, bias=False):
        super(Conv2D_Norm_Activation, self).__init__()
        pad = (kernel_size - 1) // 2  # kernel size is 3 or 0
        self.sigmoid = False
        assert norm_type in (None, 'BN', 'GN'), 'norm type just support BN or GN'
        self.norm_type = norm_type
        self.darknetConv = nn.ModuleList()
        self.darknetConv.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=pad, bias=bias))
        if norm_type == 'BN':
            self.darknetConv.add_module('bn', nn.BatchNorm2d(out_channels))
        elif norm_type == 'GN':
            if not isinstance(num_groups, int):
                num_groups = 32
            self.darknetConv.add_module('gn', nn.GroupNorm(out_channels, num_groups=num_groups))
        if activation == 'relu':
            self.darknetConv.add_module('relu', nn.ReLU(inplace=True))
        elif activation == 'leaky':
            self.darknetConv.add_module('leaky', nn.LeakyReLU(0.1, inplace=True))
        elif activation == 'swish':
            self.darknetConv.add_module('swish', Swish())
        elif activation == 'mish':
            self.darknetConv.add_module('mish', Mish())
        elif activation == 'logistic':
            self.darknetConv.add_module('logistic', torch.nn.Sigmoid())
        else:
            pass

    def forward(self, x):
        for dc in self.darknetConv:
            x = dc(x)
        return x


class CoordConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, with_r=False, activation='leaky', norm_type='BN', num_groups=None, coord_conv=True):
        super().__init__()
        self.coord_conv = coord_conv
        if self.coord_conv:
            self.addcoords = AddCoords(with_r=with_r)
            # in_size = in_channels
            in_channels += 2
            if with_r:
                in_channels += 1
        self.conv = Conv2D_Norm_Activation(in_channels, out_channels, kernel_size=kernel_size, activation=activation, norm_type=norm_type,
                                           num_groups=num_groups)

    def forward(self, x):
        if self.coord_conv:
            x = self.addcoords(x)
        # ret = x
        x = self.conv(x)
        return x


if __name__ == '__main__':
    x = torch.ones([4, 24, 200, 200])
    Add = AddCoords()
    y = Add(x)
