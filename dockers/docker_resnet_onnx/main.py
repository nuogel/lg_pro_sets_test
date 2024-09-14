import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from resnet_onnx_forward import RESNET
from starlette.middleware.cors import CORSMiddleware
import uvicorn
import time

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 设置允许的origins来源
    allow_credentials=False,
    allow_methods=["*"],  # 设置允许跨域的http方法，比如 get、post、put等。
    allow_headers=["*"])


@app.get("/liveness")
async def liveness():
    return "ok"


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), ):
    img_bin = await file.read()
    nparr = np.fromstring(img_bin, np.uint8)
    img_decode = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    start = time.time()
    result = resnet.forward(img_decode)
    print("cost time: ", "%.2f" % (time.time() - start), "s")
    return result


if __name__ == '__main__':
    onnxpath = 'resnet_sleep.pth.onnx_sim'
    resnet = RESNET(onnxpath)
    uvicorn.run(app=app,
                host='0.0.0.0',
                port=8080,
                workers=0)
