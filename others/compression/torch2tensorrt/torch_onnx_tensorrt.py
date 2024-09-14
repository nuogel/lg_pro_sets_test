import tensorrt
import torch
# from torch2trt import torch2trt, TRTModule
import onnx
from onnxsim import simplify
import onnxruntime
import cv2
import numpy as np
import onnx_tensorrt.backend as backend
import time
import sys
sys.path.append('/home/dell/lg/code/lg_pro_sets')
print(tensorrt.__version__)


def get_img_np_nchw(filename=None):
    if filename == None:
        filename = '/media/dell/data/voc/VOCdevkit/VOC2007/JPEGImages/000005.jpg'
    image = cv2.imread(filename)
    image_cv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_cv = cv2.resize(image_cv, (640, 640))
    miu = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = np.array(image_cv, dtype=float) / 255.
    r = (img_np[:, :, 0] - miu[0]) / std[0]
    g = (img_np[:, :, 1] - miu[1]) / std[1]
    b = (img_np[:, :, 2] - miu[2]) / std[2]
    img_np_t = np.array([r, g, b])
    img_np_nchw = np.expand_dims(img_np_t, axis=0)
    return np.asarray(img_np_nchw, dtype=np.float32)


def pytorchmodel(imgdata, cuda=False, times=1):
    model_path = 'tmp/yolov5_with_model.pth'
    model = torch.load(model_path)
    model.eval()
    if cuda:
        t0 = time.time()
        for i in range(times):
            print(i)
            # imgdata = torch.rand((1, 3, 640, 640)).cuda()
            y = model.cuda()(imgdata.cuda())
    else:
        t0 = time.time()
        for i in range(times):
            print(i)
            y = model(imgdata)
    timetorch = time.time() - t0
    print(str(times) + ' times of pytorch:', timetorch / times * 1000)
    return model, y, model_path


def _onnx_runtime(onnx_model_path, imgdata, times=1):
    sess = onnxruntime.InferenceSession(onnx_model_path)
    input_name = sess.get_inputs()[0].name
    output_name = [sess.get_outputs()[0].name, sess.get_outputs()[1].name, sess.get_outputs()[2].name]
    imgdata = np.asarray(imgdata)
    # imgdata = np.random.rand(1,3,640,640).astype(np.float32)
    t0 = time.time()
    for i in range(times):
        pred_onnx = sess.run(output_name, {input_name: imgdata})
    timetorch = time.time() - t0
    print(str(times) + ' times of onnx:', timetorch / times * 1000)
    return pred_onnx


def torch2onnx():
    imgdata = get_img_np_nchw()
    x = torch.from_numpy(imgdata)
    model, y, model_path = pytorchmodel(x, cuda=True, times=test_times)
    onnx_model_path = model_path + '.onnx'
    torch2onnx = 1
    if torch2onnx:
        torch.onnx.export(model, args=x.cuda(), f=onnx_model_path,
                          export_params=True,
                          verbose=True,
                          input_names=['img'],
                          output_names=["f1", 'f2', 'f3'],
                          enable_onnx_checker=True,
                          opset_version=11,  # default is 9, not support upsample_biliner2d...
                          do_constant_folding=False,
                          training=False)
        model = onnx.load(onnx_model_path)
        # Check that the IR is well formed
        model_simp, check = simplify(model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, onnx_model_path+'_sim')
        print('onnx:passed')
        # Print a human readable representation of the graph
        # onnx.helper.printable_graph(model.graph)

    onnx_pred = _onnx_runtime(onnx_model_path+'_sim', imgdata, times=test_times)
    dis = []
    for i, yi in enumerate(y):
        dis.append((onnx_pred[i] - y[i].cpu().detach().numpy()).max())
    print(dis)
    a = 0


def onnx2trt_with_code(onnx_model_path):  # failed
    # with code
    model = onnx.load(onnx_model_path)
    engine = backend.prepare(model, device='CUDA:0')
    imgdata = get_img_np_nchw()
    output_data = engine.run(imgdata)[0]
    print(output_data)
    print(output_data.shape)


def torch2trt_lg():
    imgdata = get_img_np_nchw()
    x = torch.from_numpy(imgdata).cuda()
    model, y, model_path = pytorchmodel(x, cuda=True, times=test_times)
    model.eval()
    convert = 1
    if convert:
        x_trt = torch.ones((1, 3, 640, 640)).cuda()
        model_trt = torch2trt(model, [x_trt])
        torch.save(model_trt.state_dict(), onnx_model_path + '.statedict_trt')

    model_trt_2 = TRTModule()
    model_trt_2.load_state_dict(torch.load(onnx_model_path + '.statedict_trt'))
    model_trt_2.eval()
    t0 = time.time()
    for i in range(test_times):
        print(i)
        # x = torch.rand((1, 3, 640, 640)).cuda()
        y_trt = model_trt_2(x)
    timetorch = time.time() - t0
    print(str(test_times) + ' times of tensorRt:', timetorch / test_times * 1000)

    dis = []
    for i, yi in enumerate(y):
        dis.append((y_trt[i] - y[i]).max())
    print(dis)


def trt_forward():
    '''
    use the state dict of trt directly ,as below:
    '''
    imgdata = get_img_np_nchw()
    x = torch.from_numpy(imgdata).cuda()
    # x = torch.rand((4, 3, 640, 640)).cuda()
    model_trt_2 = TRTModule()
    model_trt_2.load_state_dict(torch.load(onnx_model_path + '.trt'))
    y_trt = model_trt_2(x)


if __name__ == '__main__':
    test_times = 1
    onnx_model_path = 'tmp/yolov5_with_model.pth.onnx'
    # trt_forward()
    torch2onnx()
    # onnx2trt_with_code(onnx_model_path)
    torch2trt_lg()
'''
test report:
100 times of pytorch-cpu: 128.8730549812317 ms/img
100 times of onnx-cpu: 83.4130311012268 ms/img
100 times of pytorch-gpu: 26.613991260528564 ms/img
100 times of tensorRt-gpu: 0.25597095489501953 ms/img
'''
