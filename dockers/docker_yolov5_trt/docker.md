docker build -t yolov5:v0 -f Dockerfile .

docker run \
--gpus all \
--rm \
-p 8080:80 \
-p 8089:8089 \
-e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility \
--privileged=true \
-it yolov5:v0 bash

## 不能向上获取父目录
ADD .. /workspace/code
可以采用Docker Compose的方式
