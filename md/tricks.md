# --------PIP tricks---------
##  linux pip install
```
pip show numpy

```

the default packages location of pip install is '/home/lg/.local/lib/python3.6/'


##  pip commands
```
pip install *** /
-i https://pypi.tuna.tsinghua.edu.cn/simple

pip install opencv-python
pip install torch torchvision
```

when pip reinstall the packages that already exist:\
```
python -m pip install --upgrade pip
```
might solve that problem.

## pip  source
```
Linux下，修改 ~/.pip/pip.conf
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
[install]
trusted-host=mirrors.aliyun.com

windows下，在user目录中创建一个pip目录，如：C:\Users\xx\pip，新建文件pip.ini。内容同上。
```

##  linux pip install
```
pip show numpy

```

the default packages location of pip install is '/home/lg/.local/lib/python3.6/'


##  pip install opencv3

pip install opencv

##  delete nvidia-driver
```
sudo apt-get --purge remove nvidia*
sudo apt autoremove

To remove NVIDIA Drivers:
$ sudo apt-get --purge remove "*nvidia*"

To remove CUDA Toolkit:
$ sudo apt-get --purge remove "*cublas*" "cuda*"
```


#--------Torch Tricks--------
## import 
mod = importlib.import_module('numpy') \
means: import mumpy

## sort() and natsort()
```
from natsort import natsorted
a = ['1.mp4', '3.mp4', '10.mp4', '2.mp4']
b = natsorted(a)
print(b)

```

## 多线程处理数据
```
from concurrent.futures import ProcessPoolExecutor
from functools import partial

executor = ProcessPoolExecutor(max_workers=num_workers)
executor.submit(partial(fun, out_dir, index, wav_path, text)))

def fun(out_dir, index, wav_path, text):
    ...
```
## 创建并开启子进程
```python
from multiprocessing import Process
import time
import random
def func(name):
    print('%s' %name)
    time.sleep(random.randrange(1,5))
    print('%s  end' %name)
p1=Process(target=func,args=('name1',)) #必须加,号
p2=Process(target=func,args=('name2',))
p3=Process(target=func,args=('name3',))
p4=Process(target=func,args=('name4',))

p1.start()
p2.start()
p3.start()
p4.start()

#有的人会有疑问:既然join是等待进程结束,那么我像下面这样写,进程不就又变成串行的了吗?
#当然不是了,必须明确：p.join()是让谁等？
#很明显p.join()是让主线程等待p的结束，卡住的是主线程而绝非进程p，

#详细解析如下：
#进程只要start就会在开始运行了,所以p1-p4.start()时,系统中已经有四个并发的进程了
#而我们p1.join()是在等p1结束,没错p1只要不结束主线程就会一直卡在原地,这也是问题的关键
#join是让主线程等,而p1-p4仍然是并发执行的,p1.join的时候,其余p2,p3,p4仍然在运行,等#p1.join结束,可能p2,p3,p4早已经结束了,这样p2.join,p3.join.p4.join直接通过检测，无需等待
# 所以4个join花费的总时间仍然是耗费时间最长的那个进程运行的时间
p1.join()
p2.join()
p3.join()
p4.join()

print('主线程')


#上述启动进程与join进程可以简写为
 p_l=[p1,p2,p3,p4]
 
 for p in p_l:
     p.start()
 
 for p in p_l:
     p.join()

```
## 语音特征
    import audio

    wav = audio.load_wav(wav_path)

    if hparams.rescaling:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max

    # Compute the linear-scale spectrogram from the wav:
    spectrogram = audio.spectrogram(wav).astype(np.float32)
    n_frames = spectrogram.shape[1]

    # Compute a mel-scale spectrogram from the wav:
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)
