import numpy as np

class VAD:
    def __init__(self):
        # 初始短时能量高门限
        self.amp1 = 140
        # 初始短时能量低门限
        self.amp2 = 120
        # 初始短时过零率高门限
        self.zcr1 = 10
        # 初始短时过零率低门限
        self.zcr2 = 5
        # 允许最大静音长度
        self.maxsilence = 60
        # 语音的最短长度
        self.minlen = 30
        # 能量最大值
        # self.max_en = 20000

        self.frame_len = 256
        self.frame_inc = 128
        # 初始状态为静音
        self.status = 0
        self.count = 0
        self.silence = 0
        self.cur_status = 0

    def _load_wav(self, frame):
        cache_frames = []
        while len(frame) > self.frame_len:
            frame_block = frame[:self.frame_len]
            cache_frames.append(frame_block)
            frame = frame[self.frame_len:]
        else:
            cache_frames.append(frame)
        return cache_frames


    # 需要添加录音互斥功能能,某些功能开启的时候录音暂时关闭
    def ZCR(self, curFrame):
        # 过零率
        tmp1 = curFrame[:-1]
        tmp2 = curFrame[1:]
        sings = (tmp1 * tmp2 <= 0)
        diffs = (tmp1 - tmp2) > 0.02
        zcr = np.sum(sings * diffs)
        return zcr

    def STE(self, curFrame):
        # 短时能量
        amp = np.sum(np.abs(curFrame))
        return amp

    def speech_status(self, curFrame, max_en):
        curFrame = curFrame / max_en
        amp = self.STE(curFrame) ** 2
        zcr = self.ZCR(curFrame)
        status = 0
        # 0= 静音， 1= 可能开始 , 2  语音段, 3 結束
        if self.cur_status in [0, 1, 3]:
            # 确定进入语音段
            if amp > self.amp1:
                status = 2
                self.silence = 0
                self.count += 1
            # 可能处于语音段
            elif amp > self.amp2 or zcr > self.zcr2:
                status = 1
                self.count += 1
            # 静音状态
            else:
                status = 0
                self.count = 0
                self.count = 0
        # 2 = 语音段
        elif self.cur_status == 2:
            # 保持在语音段
            if amp > self.amp2 or zcr > self.zcr2:
                self.count += 1
                status = 2
            # 语音将结束
            else:
                # 静音还不够长，尚未结束
                self.silence += 1
                if self.silence < self.maxsilence:
                    self.count += 1
                    status = 2
                # 语音长度太短认为是噪声
                elif self.count < self.minlen:
                    status = 0
                    self.silence = 0
                    self.count = 0
                # 语音结束
                else:
                    status = 3
                    self.silence = 0
                    self.count = 0
        # print(status)
        self.cur_status = status

        return status

    pass