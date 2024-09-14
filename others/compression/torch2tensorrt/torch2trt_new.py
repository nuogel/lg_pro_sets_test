import torch
from torch2trt import torch2trt
from torch2trt import TRTModule

# create some regular pytorch model...

model = torch.load('yolov5_with_model.pth').eval().cuda()
# create example data
x = torch.ones((1, 3, 640, 640)).cuda()

y = model(x)

# convert to TensorRT feeding sample data as input
model_trt = torch2trt(model, [x])

y = model(x)
y_trt = model_trt(x)

# check the output against PyTorch
print(torch.max(torch.abs(y - y_trt)))


torch.save(model_trt.state_dict(), 'tmp/alexnet_trt.pth')

from torch2trt import TRTModule

model_trt = TRTModule()

model_trt.load_state_dict(torch.load('alexnet_trt.pth'))

y_trt = model_trt(x)
print(torch.max(torch.abs(y - y_trt)))
