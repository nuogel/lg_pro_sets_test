import os
import onnxruntime
from onnxruntime import SessionOptions
import numpy as np
import time
import cv2
import torch
from torchvision import transforms
import math

def letter_box(img, label, data_info={}, new_shape=(416, 416), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    pad_w, pad_h = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        pad_w, pad_h = np.mod(pad_w, 64), np.mod(pad_h, 64)  # wh padding
    elif scaleFill:  # stretch
        pad_w, pad_h = 0.0, 0.0
        new_unpad = new_shape
        ratio = [new_shape[0] / shape[1], new_shape[1] / shape[0]]  # width, height ratios

    pad_w /= 2  # divide padding into 2 sides
    pad_h /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    if label != []:
        labels = label.copy()
        labels[:, 1] = ratio[0] * label[:, 1] + pad_w
        labels[:, 2] = ratio[1] * label[:, 2] + pad_h
        labels[:, 3] = ratio[0] * label[:, 3] + pad_w
        labels[:, 4] = ratio[1] * label[:, 4] + pad_h
    else:
        labels = label
    data_info['img_raw_size(h,w)'] = shape
    data_info['ratio(w,h)'] = np.asarray(ratio)  # new/old
    data_info['padding(w,h)'] = np.asarray([pad_w, pad_h])
    data_info['size_now(h,w)'] = np.asarray(img.shape[:2])

    return img, labels, data_info


def transpose(img, label=None):
    mean = [0.485, 0.456, 0.406]  # RGB
    std = [0.229, 0.224, 0.225]
    mean = np.asarray(mean, np.float32)
    std = np.asarray(std, np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img, dtype=np.float32) / 255.
    # img = (img - self.cfg.mean) / self.cfg.std
    img = torch.from_numpy(img).permute(2, 0, 1)  # C, H, W
    img = transforms.Normalize(mean, std, inplace=True)(img)
    if label is None:
        return img, label
    if isinstance(label, list):
        label = np.asarray(label, np.float32)

    label = np.insert(label, 0, 0, 1)
    label = torch.from_numpy(label).to(torch.float32)
    return img, label


class RESNET:
    def __init__(self, onnxpath=None):
        self.inputshape = (224, 224)
        options = SessionOptions()
        options.intra_op_num_threads = 1
        self.sess = onnxruntime.InferenceSession(onnxpath, providers=['CUDAExecutionProvider'], sess_options=options)
        print('onnx using:', self.sess.get_providers())
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = [self.sess.get_outputs()[0].name]

    def preprocess(self, img):
        img, label, data_info = letter_box(img, label=[], new_shape=self.inputshape, auto=False, scaleup=True)
        img, label = transpose(img, None)
        img = torch.unsqueeze(img, 0)
        return img, label, data_info

    def forward(self, img):
        model_output = {"entity_struct": []}
        img_input, labels, data_info = self.preprocess(img)
        predicts = self.sess.run(self.output_name, {self.input_name: np.asarray(img_input)})
        sigmod_out = 1/(1+np.exp(-predicts[0][0]))
        out = int(np.argmax(sigmod_out))
        score = float(sigmod_out[out])
        dict_sleep = {0: '否', 1: '是'}
        item = {
            "name": '睡岗',
            "parent_id": -1,
            "desc": "",
            "bndbox": 'none',
            "confidence": score,
            "extral": "",
            "property": [
                {
                    "name": '睡岗',
                    "value": dict_sleep[out],
                    "desc": "",
                    "confidence": score,
                    "extral": ""
                }
            ]
        }
        model_output["entity_struct"].append(item)
        return model_output


def main():
    onnxpath = '/home/dell/lg/code/lg_pro_sets/dockers/docker_resnet_onnx/resnet_sleep.pth.onnx_sim'
    imgp = '/home/dell/lg/code/lg_pro_sets/dockers/docker_resnet_onnx/testimgs'
    resnet = RESNET(onnxpath)
    imgps = os.listdir(imgp)
    y = 0
    n = 0
    time0 = time.time()
    for imgp_i in imgps:
        img = cv2.imread(os.path.join(imgp, imgp_i))
        res = resnet.forward(img)
        out = res['entity_struct'][0]['property'][0]['value']
        if out=='是':
            y += 1
        else:
            n += 1
    print('y', y / len(imgps))
    print('n', n / len(imgps))
    timeall = time.time() - time0
    print('time cost:', timeall / len(imgps))


# CPU速度：0.015s/img
# ONNX gpu time cost: 0.005s/img

if __name__ == '__main__':
    main()
