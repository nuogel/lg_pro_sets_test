'''
Parse the prediction from SR model.
'''

import cv2
import torch


def parse_Tensor_img(tensor_imgs, pixcels_norm=[177.0, 1.], save_paths=None, show_time=20000):
    for i in range(tensor_imgs.shape[0]):
        img = tensor_imgs[i]
        img = img.mul(pixcels_norm[1]).add(pixcels_norm[0]).clamp(0, 255).round().to('cpu', torch.uint8).numpy()
        RGB2BGR = 0
        if RGB2BGR:
            img_cat_copy = img.copy()
            img[..., 0] = img_cat_copy[..., 2]
            img[..., 2] = img_cat_copy[..., 0]
        if save_paths:
            cv2.imwrite(save_paths[i], img)
            print('     ---->', save_paths[i])
        if show_time:
            cv2.imshow('img', img)
            cv2.waitKey(show_time)
