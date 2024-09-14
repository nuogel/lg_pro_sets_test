import torch.nn as nn
import torch
from torch.autograd import Variable
import os
import cv2
import numpy as np
import glob


def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()

    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = Variable(grid) + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid, padding_mode='border')
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid, padding_mode='border')

    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return output * mask


def load_img_flo(path='E:\datasets\FlyingChairs_release\data'):
    for flow_map in sorted(glob.glob(os.path.join(path, '*_flow.flo'))):
        flow_path = os.path.basename(flow_map)
        root_filename = flow_path[:-9]
        image1_path = os.path.join(path, root_filename + '_img1.ppm')
        image2_path = os.path.join(path, root_filename + '_img2.ppm')
        if not (os.path.isfile(os.path.join(path, image1_path)) and os.path.isfile(os.path.join(path, image2_path))):
            continue

        img1 = cv2.imread(image1_path)
        img2 = cv2.imread(image2_path)
        flo = load_flo(flow_map)

        img2_tensor = torch.from_numpy(img2).type(torch.float).permute(2, 0, 1).unsqueeze(0)
        flo = torch.from_numpy(flo).permute(2, 0, 1).unsqueeze(0)

        img2_wap = warp(img2_tensor, flo).squeeze().permute(1, 2, 0)
        img2_wap = np.asarray(img2_wap, dtype=np.uint8)

        img = np.concatenate((img1, img2, img2_wap), axis=1)

        cv2.imshow('img', img)
        cv2.waitKey()


def load_flo(path):
    with open(path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert (202021.25 == magic), 'Magic number incorrect. Invalid .flo file'
        h = np.fromfile(f, np.int32, count=1)[0]
        w = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2 * w * h)
    # Reshape data into 3D array (columns, rows, bands)
    data2D = np.resize(data, (w, h, 2))
    return data2D


if __name__ == '__main__':
    load_img_flo()
