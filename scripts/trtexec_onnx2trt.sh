/media/dell/data/installpkgs/TensorRT-8.0.3.4.Linux.x86_64-gnu.cuda-10.2.cudnn8.2/TensorRT-8.0.3.4/targets/x86_64-linux-gnu/bin/trtexec \
--explicitBatch \
--onnx=/home/dell/lg/code/lg_pro_sets/others/model_compression/torch2tensorrt/tmp/yolov5_with_model.pth.onnx \
--saveEngine=yolov5_with_model.trt_engine \
#--fp16 \
--workspace=10240 \
--verbose
