# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:21.10-py3

WORKDIR /workspace

ADD . /workspace/code

'''
#e.g:高空抛物demo.
# FROM ecpe4s/ubuntu18.04-spack
#
# WORKDIR /workspace
# ADD . /workspace/code
#
# # # ldd libchight_throw.so, 发现缺少依赖，将其添加到lib库。不能在
# ADD ./lib/libctracker.so /lib
#
# RUN apt-get install -y libsm6
# # -y参数可以跳过软件的询问，相当于回答了yes。
# RUN apt-get install -y libxrender1
# RUN apt-get install -y libxext-dev
# # Using tsinghua pipy mirror
# # RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -U pip
# # RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/
#
# RUN pip install numpy  -i https://pypi.tuna.tsinghua.edu.cn/simple
# # RUN pip install opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple
# RUN pip install opencv-python==3.4.1.15 -i https://pypi.tuna.tsinghua.edu.cn/simple
# RUN pip install opencv-contrib-python==3.4.1.15 -i https://pypi.tuna.tsinghua.edu.cn/simple
#
# WORKDIR /workspace/code
# # CMD ["python", "demo.py"]
'''
