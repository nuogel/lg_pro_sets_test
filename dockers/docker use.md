## DOCKERFILE 按需构建镜像
```
FROM nvcr.io/nvidia/pytorch:21.11-py3

WORKDIR /workspace
ADD . /workspace/code

# -y参数可以跳过软件的询问，相当于回答了yes。
RUN apt update
RUN apt install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-dev
RUN apt-get install -y libsm6 libxext6 libxrender-dev
# Using tsinghua pipy mirror
# RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -U pip
# RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/
#
RUN pip install fastapi uvicorn  python-multipart -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install numpy  -i https://pypi.tuna.tsinghua.edu.cn/simple
# RUN pip install opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install opencv-python==4.1.2.30 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install opencv-contrib-python==4.1.2.30 -i https://pypi.tuna.tsinghua.edu.cn/simple
#
#
WORKDIR /workspace/code
CMD ["python", "main.py"]

```

## docker build
```
# build镜像 - tensorrt
docker build -t rock-electricity-meter-ocr -f Dockerfile .

# 启动镜像容器，
docker run \
--gpus all \
--rm \
-p 8080:80 \
-p 8089:8089 \
-e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility \
--privileged=true \
-it rock-electricity-meter-ocr:latest bash
```

## docker start
```
docker run \
-p 8080:80 \
-p 8089:8089 \
--privileged=true \
-it throwing:v0
```

## docker push
- docker login -u 'luogeng'
- docker tag xx:v0  luogeng/xx:v0 #tag 前需加用户名
- docker push luogeng/xx:v0

## docker pull
docker pull luogeng/xx:v0

