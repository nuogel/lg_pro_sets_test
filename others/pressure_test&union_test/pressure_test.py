# coding: utf-8
# Locust  性能测试工具
import json
import uuid
# import base64
import requests
from locust import task, between, HttpUser  # User, TaskSet
import sys

sys.path.append("../config")

ip = '172.31.3.213'
# ip = '0.0.0.0'
pode = '8080'
# pode = '8000'
srv_name = 'api/test'
# srv_name = 'image/shape_file'

SRV_HOST = f'http://{ip}:{pode}/{srv_name}'
import os

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# http://srv-ai-detection-construction-truck__ai.rocktl.com/

img_path = '/media/dell/data/ocr/电表识别/1st_dataset/images/201008670484661071.jpg'

# LIVENESS_HOST = f'http://0.0.0.0:80/liveness'

with open(img_path, 'rb') as f:
    img_binary_stream = f.read()
    img_dict = dict(file=img_binary_stream)


class MyTaskSet(HttpUser):
    def on_start(self):
        # 定义引用变量
        self.img_dict = img_dict

    # wait_time: locust 定义请求间隔
    wait_time = between(0.1, 0.2)  # 模拟用户在执行每个任务之间等待的最小时间，单位为秒；

    # 定义性能测试任务集合 inference
    @task
    def sms_model_srv(self):
        dummy_traceID = str(uuid.uuid4())
        test_post_response = self.client.post(SRV_HOST, files=self.img_dict, headers={"X-B3-Traceid": dummy_traceID})
        result = json.loads(test_post_response.content.decode('utf-8'))
        assert result['code'] == 200

    # 定义性能测试任务集合 liveness
    # @task
    # def get_liveness(self):
    #     response = self.client.get(LIVENESS_HOST)
    #     expect = {SRV_NAME: 'ok'}
    #     result = response.json()
    #     assert expect == result

# bash 执行命令 locust -f test.py --host http://0.0.0.0:800 --web-host 0.0.0.0 -u 200 -r 100 --run-time 3m
# 网页打开浏览器: http://0.0.0.0:8089
# Number of total users to simulate 模拟的用户数；模拟的总虚拟用户数
# Spawn rate (users spawned/second) 每秒产生的用户数；每秒启动的虚拟用户数
