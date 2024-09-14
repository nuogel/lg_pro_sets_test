#!/usr/bin/env bash
cd ./lgdet/util/util_nms/

CUDA_PATH=/usr/local/cuda/

python3 build.py build_ext --inplace

cd ..
