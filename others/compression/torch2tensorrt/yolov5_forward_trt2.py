import os.path
import tensorrt as trt
import torch
# from torch2trt import torch2trt, TRTModule
import onnx
import threading
import onnxruntime
import cv2
import pycuda.driver as cuda
import numpy as np
import onnx_tensorrt.backend as backend
import time
import cv2
from lgdet.postprocess.parse_factory import ParsePredict
from lgdet.util.util_show_img import _show_img
from lgdet.util.util_time_stamp import Time
import torch
from lgdet.util.util_yml_parse import parse_yaml
from argparse import ArgumentParser
from lgdet.util.util_lg_transformer import LgTransformer
import sys
from lgdet.config.cfg import prepare_cfg
from lgdet.util.util_prepare_device import load_device
from common import get_engine, allocate_buffers, postprocess_the_outputs, do_inference

sys.path.append('/home/dell/lg/code/lg_pro_sets')


def _parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--type', default='OBD'  # SR_DN
                        , type=str, help='yml_path')
    parser.add_argument('--model', type=str, help='yml_path')
    parser.add_argument('--checkpoint', '--cp', default=1
                        , help='Path to the checkpoint to be loaded to the model')
    parser.add_argument('--pre_trained', '--pt', default=0  # 'saved/checkpoint/fcos_voc_77_new.pkl'
                        , help='Epoch of continue training')
    parser.add_argument('--batch_size', '--bz', default=1, type=int, help='batch size')
    parser.add_argument('--gpu', help='number works of dataloader')
    parser.add_argument('--ema', default=0, type=int, help='ema')
    parser.add_argument('--number_works', '--nw', default=0, type=int, help='number works of dataloader')
    parser.add_argument('--debug', '--d', action='store_true', default=False, help='Enable verbose info')
    parser.add_argument('--score_thresh', '--st', default=0.2, type=float, help='score_thresh')

    return parser.parse_args()


class Process:
    def __init__(self, cfg, path=None):
        self.mode = path.split('.')[-1]
        self.cfg = cfg
        self.inputshape = (640, 640)
        self.parsepredict = ParsePredict(cfg)
        self.lgtransformer = LgTransformer(cfg)

    def preprocess(self, img):
        img, label, data_info = self.lgtransformer.letter_box(img, label=[], new_shape=self.inputshape, auto=False, scaleup=True)
        img, label = self.lgtransformer.transpose(img, None)
        img = torch.unsqueeze(img, 0)
        return img, label, data_info

    def postprocess(self, predicts, img_input, img_raw, data_info):
        labels_pres = self.parsepredict.parse_predict(predicts)
        labelsp = self.parsepredict.predict2labels(labels_pres, [data_info])
        _show_img([img_raw], labelsp, img_in=img_input, cfg=self.cfg, is_training=False, relative_labels=False)


if __name__ == '__main__':
    score = False
    torchpath = '/home/dell/lg/code/lg_pro_sets/others/model_compression/torch2tensorrt/tmp/yolov5_with_model.pth'
    onnxpath = '/home/dell/lg/code/lg_pro_sets/others/model_compression/torch2tensorrt/tmp/yolov5_with_model.pth.onnx'
    torch2trtpath = '/home/dell/lg/code/lg_pro_sets/others/model_compression/torch2tensorrt/tmp/yolov5_with_model.pth.onnx.statedict_trt'
    onnx2trt32path = '/home/dell/lg/code/lg_pro_sets/others/model_compression/torch2tensorrt/tmp/yolov5_with_model.pth.onnx.trt_fp32'
    onnx2trt16path = '/home/dell/lg/code/lg_pro_sets/others/model_compression/torch2tensorrt/tmp/yolov5_with_model.pth.onnx.trt_fp16'
    imgp = '/media/dell/data/voc/VOCdevkit/VOC2007/JPEGImages/'
    args = _parse_arguments()
    cfg = parse_yaml(args)
    cfg, args = prepare_cfg(cfg, args, is_training=False)
    load_device(cfg)
    max_batch_size = 1
    shape_of_outputs = [(max_batch_size, 75, 80, 80), (max_batch_size, 75, 40, 40), (max_batch_size, 75, 20, 20), ]
    prcess = Process(cfg, onnx2trt32path)

    # engine = get_engine(engine_file_path=onnx2trt32path)
    # context = engine.create_execution_context()
    with get_engine(engine_file_path=onnx2trt32path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = allocate_buffers(engine)  # input, output: host # bindings
        for imgp_i in os.listdir(imgp):
            img = cv2.imread(os.path.join(imgp, imgp_i))
            img_raw = img.copy()
            img_input, labels, data_info = prcess.preprocess(img)
            img = np.asarray(img_input, dtype=np.float32)
            inputs[0].host = img.reshape(-1)
            trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)  # numpy data
            predicts = []
            for trt_output, shape_of_output in zip(trt_outputs, shape_of_outputs):
                trt_out = postprocess_the_outputs(trt_output, shape_of_output)
                predicts.append(torch.from_numpy(trt_out).cuda())
            prcess.postprocess(predicts, img_input, img_raw, data_info)
