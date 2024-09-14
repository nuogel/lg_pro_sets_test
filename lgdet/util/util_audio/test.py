# -*- coding: utf-8 -*-
import pyaudio
import wave
import librosa
import numpy as np

def play():
    chunk = 2048  # 2014kb
    wf = wave.open("/home/mao/lgws/0_010044.wav")
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()), channels=wf.getnchannels(),
                    rate=wf.getframerate(), output=True)
    stream = p.open(format=8, channels=1, rate=16000, output=True)

    data = wf.readframes(chunk)  # 读取数据
    print(data)
    while data != []:  # 播放
        data = np.frombuffer(data, dtype="int8")
        stream.write(data)
        data = wf.readframes(chunk)

        print('while循环中！')
        print(data)
    stream.stop_stream()  # 停止数据流
    stream.close()
    p.terminate()  # 关闭 PyAudio
    print('play函数结束！')


file_path = '/home/mao/lgws/0_010044.wav'
wav = np.asarray(librosa.load(file_path, sr=16000)[0]*32767, dtype=np.int16)
p = pyaudio.PyAudio()
stream = p.open(format=8, channels=1, rate=16000, output=True)
lenth = 256
data = wav[0:lenth]
i=0
while data!=[]:
    data = bytes(data)
    stream.write(data)
    i+=1
    data = wav[lenth*i:lenth*(i+1)]

play()

# 停止数据流
stream.stop_stream()
stream.close()

# 关闭 PyAudio
p.terminate()