FROM  nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04

WORKDIR /workspace
ADD . /workspace/code

RUN apt update
RUN apt-get install -y  zlib1g-dev libbz2-dev libssl-dev libncurses5-dev libsqlite3-dev wget python3-distutils
RUN apt-get install -y python3.8
WORKDIR /usr/lib/python3.8
RUN apt-get install -y python3-distutils
RUN wget https://bootstrap.pypa.io/get-pip.py  --no-check-certificate
RUN python3.8 get-pip.py

WORKDIR /workspace/code

RUN pip3 install fastapi -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install onnxruntime-gpu==1.5.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install onnxruntime-tools==1.5.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install opencv-python==3.4.11.41 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install opencv-contrib-python==3.4.11.41 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install --pre torch torchvision  --extra-index-url https://download.pytorch.org/whl/nightly/cu102
RUN pip3 install vim uvicorn python-multipart -i https://pypi.tuna.tsinghua.edu.cn/simple


#
RUN apt install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-dev
RUN apt-get install -y libsm6
RUN apt-get install -y libxrender1
RUN apt-get install -y libxext-dev


