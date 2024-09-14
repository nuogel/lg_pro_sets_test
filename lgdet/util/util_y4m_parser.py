# coding=utf-8
# 将该文件放到hr中，和y4m文件同级。
# 运行后，每个y4m文件会在hr/test_temp中生成一个文件夹，文件夹中是各帧的png文件。
from pathlib import Path
import subprocess as sbp
import os
'''
## 初赛训练数据下载链接
round1_train_input:
http://tianchi-media.oss-cn-beijing.aliyuncs.com/231711_youku/round1/train/input/youku_00000_00049_l.zip
http://tianchi-media.oss-cn-beijing.aliyuncs.com/231711_youku/round1/train/input/youku_00050_00099_l.zip
http://tianchi-media.oss-cn-beijing.aliyuncs.com/231711_youku/round1/train/input/youku_00100_00149_l.zip

round1_train_label:
http://tianchi-media.oss-cn-beijing.aliyuncs.com/231711_youku/round1/train/label/youku_00000_00049_h_GT.zip
http://tianchi-media.oss-cn-beijing.aliyuncs.com/231711_youku/round1/train/label/youku_00050_00099_h_GT.zip
http://tianchi-media.oss-cn-beijing.aliyuncs.com/231711_youku/round1/train/label/youku_00100_00149_h_GT.zip

## 初赛验证数据下载链接
round1_val_input:
http://tianchi-media.oss-cn-beijing.aliyuncs.com/231711_youku/round1/train/input/youku_00150_00199_l.zip

round1_val_label:
http://tianchi-media.oss-cn-beijing.aliyuncs.com/231711_youku/round1/train/label/youku_00150_00199_h_GT.zip

## 初赛测试数据下载链接
round1_test_input:
http://tianchi-media.oss-cn-beijing.aliyuncs.com/231711_youku/round1/test/input/youku_00200_00249_l.zip
————————————————
版权声明：本文为CSDN博主「三寸光阴___」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_38109843/article/details/93488207
'''
path = 'E:/datasets/youku/youku_00200_00249_h_GT/'
for p in Path(path).iterdir():  # 文件夹迭代器
    if p.suffix == '.y4m':  # 如果pp是y4m文件，则进行转换操作
        print(str(p))
        # sbp.run(['mkdir',f'{p.stem}'])
        if not os.path.isdir(path+p.stem):
            os.mkdir(path+p.stem)
        print(path+p.stem)  # Youku_00000_l
        write_path = os.path.join(path+p.stem, )
        sbp.run(['ffmpeg', '-i', f'{p}', '-vsync', '0', f'{write_path}/{p.stem}_%3d.png', '-y'])
