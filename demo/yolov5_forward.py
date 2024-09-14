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
from others.compression.torch2tensorrt.common import get_engine, allocate_buffers, postprocess_the_outputs, do_inference

sys.path.append('/home/dell/lg/code/lg_pro_sets')


def _parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--type', default='obd'  # SR_DN
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


class YOLOV5:
    def __init__(self, cfg, path=None):
        self.mode = path.split('.')[-1]
        self.cfg = cfg
        self.inputshape = (640, 640)
        self.parsepredict = ParsePredict(cfg)
        self.lgtransformer = LgTransformer(cfg)
        self.apolloclass2num = dict(zip(self.cfg.TRAIN.CLASSES, range(len(self.cfg.TRAIN.CLASSES))))

        if self.mode == 'pth':
            self.model_torch = torch.load(path)
        elif self.mode == 'onnx':
            self.sess = onnxruntime.InferenceSession(onnxpath)
            self.input_name = self.sess.get_inputs()[0].name
            self.output_name = [self.sess.get_outputs()[0].name, self.sess.get_outputs()[1].name, self.sess.get_outputs()[2].name]
        elif self.mode in ['trt_fp16', 'trt_fp32']:
            max_batch_size = 1
            self.engine = get_engine(engine_file_path=path)
            self.context = self.engine.create_execution_context()
            self.inputs, self.outputs, self.bindings = allocate_buffers(self.engine)  # input, output: host # bindings
            self.shape_of_outputs = [(max_batch_size, 75, 80, 80), (max_batch_size, 75, 40, 40), (max_batch_size, 75, 20, 20), ]

    def preprocess(self, img):
        img, label, data_info = self.lgtransformer.letter_box(img, label=[], new_shape=self.inputshape, auto=False, scaleup=True)
        img, label = self.lgtransformer.transpose(img, None)
        img = torch.unsqueeze(img, 0)
        return img, label, data_info

    def postprocess(self, predicts, img_input, data_info):
        labels_pres = self.parsepredict.parse_predict(predicts)
        labelsp = self.parsepredict.predict2labels(labels_pres, [data_info])
        img_raw = [img]
        _show_img(img_raw, labelsp, img_in=img_input, cfg=self.cfg, is_training=False, relative_labels=False)

    def forward(self, img):
        img_input, labels, data_info = self.preprocess(img)
        if self.mode == 'pth':
            predicts = self.forward_torch(img_input)
        elif self.mode == 'onnx':
            predicts = self.forward_onnx(img_input)
        elif self.mode in ['trt_fp16', 'trt_fp32']:
            predicts = self.forward_onnx2trt(img_input)
        else:
            print('error mode path...')
        self.postprocess(predicts, img_input, data_info)

    def forward_torch(self, img):
        predicts = self.model_torch.forward(img.cuda())
        return predicts

    def forward_torch2trt(self, img_input):
        img_input = img_input.cuda()
        predicts = self.model_trt_2.forward(img_input)
        return predicts

    def forward_onnx2trt(self, img):
        img = np.asarray(img, dtype=np.float32)
        self.inputs, self.outputs, self.bindings = allocate_buffers(self.engine)  # input, output: host # bindings
        self.stream = cuda.Stream()
        self.inputs[0].host = img.reshape(-1)
        trt_outputs = do_inference(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)  # numpy data
        predicts = []
        for trt_output, shape_of_output in zip(trt_outputs, self.shape_of_outputs):
            trt_out = postprocess_the_outputs(trt_output, shape_of_output)
            predicts.append(torch.from_numpy(trt_out).cuda())
        return predicts

    def forward_onnx(self, img):
        img_input, labels, data_info = self.preprocess(img)
        predicts = self.sess.run(self.output_name, {self.input_name: np.asarray(img_input)})
        predicts_torch = []
        for predict in predicts:
            predicts_torch.append(torch.from_numpy(predict).cuda())
        self.postprocess(predicts_torch, img_input.cuda(), data_info)


def main():
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

    yolov5 = YOLOV5(cfg, torchpath)
    time0 = time.time()
    for imgp_i in os.listdir(imgp):
        img = cv2.imread(os.path.join(imgp, imgp_i))
        yolov5.forward(img)
    timeall = time.time() - time0
    print('time cost:', timeall)


if __name__ == '__main__':
    main()
