import os
import requests
import json
import argparse
import tqdm
import time

def test_img_file(img_path,url):
    """Test image file .

        Args:
            img_path (str): 图片路径.
            url: 算法服务调用接口访问url.
        Returns:
            返回json格式识别结果
        """

    with open(img_path, 'rb') as f:
        img_binary_stream = f.read()
        img_dict = dict(file=img_binary_stream)
    rec = requests.post(url, files=img_dict)
    json_res = json.loads(rec.text)
    return json_res


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='测试')
    parser.add_argument('--image_path', type=str, default="/media/dell/data/sleep/睡岗-Corp/是",help="图片路径")
    parser.add_argument('--server_url', type=str, default="http://0.0.0.0:8080/analyze", help="服务url")
    args = parser.parse_args()
    y = 0
    n = 0
    imgps=os.listdir(args.image_path)
    time0 = time.time()
    for img_i in tqdm.tqdm(imgps):
        imgp = os.path.join(args.image_path, img_i)
        result = test_img_file(imgp, args.server_url)
        pre_lab = result['entity_struct'][0]['property'][0]['value']
        if pre_lab == '是':
            y += 1
        else:
            n += 1
    print('y:', y / len(imgps))
    print('n:', n / len(imgps))
    timeall = time.time() - time0
    print('time cost:', timeall / len(imgps))
