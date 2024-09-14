import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
from torch_onnx_tensorrt import pytorchmodel, get_img_np_nchw
import os
import torch
import time

TRT_LOGGER = trt.Logger()  # This logger is required to build an engine


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """Within this context, host_mom means the cpu memory and device means the GPU memory
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def get_engine(max_batch_size=1, onnx_file_path="", engine_file_path="", fp16_mode=False, int8_mode=False,
               save_engine=True,
               ):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine(max_batch_size, save_engine):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with trt.Builder(TRT_LOGGER) as builder, \
                builder.create_network(EXPLICIT_BATCH) as network, \
                trt.OnnxParser(network, TRT_LOGGER) as parser:

            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 30
            if int8_mode:
                assert (builder.platform_has_fast_int8 == True), "not support int8"
                builder.int8_mode = True
                builder.int8_calibrator = None  # LG
            elif fp16_mode:
                assert (builder.platform_has_fast_fp16 == True), "not support fp16"
                builder.fp16_mode = True

            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                if not parser.parse(model.read()):
                    print('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            # last_layer = network.get_layer(network.num_layers - 1)
            # network.mark_output(last_layer.get_output(0))
            last_layer = network.get_layer(network.num_layers - 1)
            # Check if last layer recognizes it's output
            if not last_layer.get_output(0):
                # If not, then mark the output using TensorRT API
                network.mark_output(last_layer.get_output(0))
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))

            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            if save_engine:
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
            return engine

    # pdb.set_trace()
    if os.path.exists(engine_file_path):
        # If a serialized engine exists, load it instead of building a new one.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine(max_batch_size, save_engine)


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer data from CPU to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def postprocess_the_outputs(h_outputs, shape_of_output):
    h_outputs = h_outputs.reshape(*shape_of_output)
    return h_outputs


def main():
    filename = '/media/dell/data/voc/VOCdevkit/VOC2007/trainval/JPEGImages/000005.jpg'
    max_batch_size = 1
    onnx_model_path = 'tmp/yolov5_with_model.pth.onnx'
    # These two modes are dependent on hardwares
    fp16_mode = False
    int8_mode = False
    trt_engine_path = 'tmp/yolov5_with_model.pth.onnx.trt_fp32'
    # Build an engine
    engine = get_engine(max_batch_size, onnx_model_path, trt_engine_path, fp16_mode, int8_mode)
    stream = cuda.Stream()

    # Create the context for this engine
    context = engine.create_execution_context()
    # Allocate buffers for input and output
    inputs, outputs, bindings = allocate_buffers(engine)  # input, output: host # bindings
    start = time.time()

    # Do inference
    img_np_nchw = get_img_np_nchw(filename)
    img_np_nchw = img_np_nchw.astype(dtype=np.float32)

    # img_np_nchw = np.random.rand(1, 3, 640, 640)

    shape_of_outputs = [(max_batch_size, 75, 80, 80), (max_batch_size, 75, 40, 40), (max_batch_size, 75, 20, 20), ]
    # Load data to the buffer
    inputs[0].host = img_np_nchw.reshape(-1)

    # inputs[1].host = ... for multiple input
    t1 = time.time()
    trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)  # numpy data
    t2 = time.time()
    print('1 time of trt:', t2 - t1)
    trt_pred = []
    for trt_output, shape_of_output in zip(trt_outputs, shape_of_outputs):
        trt_pred.append(postprocess_the_outputs(trt_output, shape_of_output))

    model, y, model_path = pytorchmodel(imgdata=torch.from_numpy(get_img_np_nchw(filename)).cuda(), cuda=True)
    dis = []
    for i, yi in enumerate(y):
        dis.append((trt_pred[i] - y[i].cpu().detach().numpy()).max())
    print(dis)
    a = 0


if __name__ == '__main__':
    main()
