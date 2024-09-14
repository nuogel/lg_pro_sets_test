import os.path
import numpy as np
import cv2
import requests
import json
import uuid


def make_img_str():
    imgpath = os.path.join(os.path.dirname(__file__), '../../', 'datasets/e.g/000005.jpg')
    img = cv2.imread(imgpath)
    rec, imgencode = cv2.imencode('.jpg', img)
    data_encode = np.array(imgencode)
    str_encode = data_encode.tobytes()

    nparr = np.fromstring(str_encode, np.uint8)
    img_decode = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    shape = img_decode.shape

    return str_encode


def test_function():
    host = '127.0.0.1:8000'
    imgpath = os.path.join(os.path.dirname(__file__), '../../', 'datasets/e.g/000005.jpg')
    imgfile = open(imgpath, 'rb')
    image_str = make_img_str()
    data = {'file': imgfile}
    url_get = [f"http://{host}/api_route/",
               f"http://{host}/path/int/{2}/",
               f"http://{host}/function/log/{10}/",
               f"http://{host}/function/sum/?a=2&b=3",
               ]
    url_post = [f"http://{host}/image/shape_file",
                ]

    test_get_response = requests.get(url_get[-1])
    if test_get_response.status_code == 200:
        print(json.loads(test_get_response.content.decode('utf-8')))
    else:
        print(test_get_response)

    otherhost = 'http://172.31.3.213:8080/api/test'
    dummy_traceID = str(uuid.uuid4())
    test_post_response = requests.post(url=otherhost, files=data, headers={"X-B3-Traceid": dummy_traceID})
    if test_post_response.status_code == 200:
        print(json.loads(test_post_response.content.decode('utf-8')))
    else:
        print(test_post_response)


def request_main():
    host_port = '172.31.3.213:8080'
    func = 'analyze'
    url = f"http://{host_port}/{func}"

    imgpath = os.path.join(os.path.dirname(__file__), '../../', 'datasets/e.g/000005.jpg')
    imgfile = open(imgpath, 'rb')
    data = {'file': imgfile}
    headers = {
        "token": "c0b8f5af62c92138"
    }
    test_get_response = requests.post(url=url, files=data, headers=headers)
    if test_get_response.status_code == 200:
        print(json.loads(test_get_response.content.decode('utf-8')))
    else:
        print(test_get_response)


if __name__ == '__main__':
    # test_function()
    request_main()
