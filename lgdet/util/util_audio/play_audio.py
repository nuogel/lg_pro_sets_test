# -*- coding: utf-8 -*-
import pyaudio
import wave
import librosa
import numpy as np
from threading import Thread


class PlayAudio:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=8, channels=1, rate=16000, output=True)
        self.buffer = 256
        self.buffer_wav = []
        self.thred = Thread(target=self._play)
        self.loop = False

    def refresh_wav(self, wav):
        self.buffer_wav = wav

    def play(self):
        self.loop = True
        self.thred.start()

    def stop(self):
        self.loop = False

    def _play(self):
        while self.loop:
            if len(self.buffer_wav) > 1000:
                data = self.buffer_wav[0:self.buffer]
                i = 0
                while data != []:
                    data = bytes(data)
                    self.stream.write(data)
                    i += 1
                    data = self.buffer_wav[self.buffer * i:self.buffer * (i + 1)]

    def close(self):
        # 停止数据流
        self.stream.stop_stream()
        self.stream.close()
        # 关闭 PyAudio
        self.p.terminate()


if __name__ == '__main__':
    file_path = 'wav_mag_vav.wav'
    wav = np.asarray(librosa.load(file_path, sr=16000)[0] * 32767, dtype=np.int16)
    play = PlayAudio()
    play.play()
    for i in range(200):
        wav_i = wav[: (i + 1) * len(wav) // 200]
        play.refresh_wav(wav_i)
    play.stop()
