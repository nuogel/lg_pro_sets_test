import librosa.filters
import librosa
import numpy as np
import copy
from scipy import signal
from scipy.io import wavfile
import torch
from pypinyin import lazy_pinyin, Style
import os
from matplotlib import pyplot as plt
from scipy.fftpack import fft, ifft


class Audio:
    def __init__(self, cfg, test=False):
        if not test:
            self.cfg = cfg.TRAIN
            self.label_dict = torch.load(cfg.PATH.CLASSES_PATH)
        else:
            self.cfg = cfg
            self.label_dict = torch.load('/media/lg/SSD_WorkSpace/LG/GitHub/lg_pro_sets/lgdet/dataloader/classes/pinyin_dict.pth')

    def load_wav(self, path):
        return librosa.load(path, sr=self.cfg.sample_rate)[0]

    def write_wav(self, wav, path):
        wav = self._dc_notch_filter(wav)
        wav = wav / np.abs(wav).max() * 0.999
        f1 = 0.8 * 32768 / max(0.01, np.max(np.abs(wav)))
        f2 = np.sign(wav) * np.power(np.abs(wav), 0.95)
        wav = f1 * f2
        dir = os.path.dirname(path)
        os.makedirs(dir, exist_ok=True)
        wavfile.write(path, self.cfg.sample_rate, wav.astype(np.int16))
        print('saved waves to: ', path)

    def preemphasize(self, wav, preemphasis=0.97):
        return signal.lfilter([1, -preemphasis], [1], wav)

    def de_emphasize(self, wav, preemphasis=0.97):
        return signal.lfilter([1], [1, -preemphasis], wav)

    def trim_silence(self, wav):
        return librosa.effects.trim(wav, top_db=self.cfg.trim_top_db, frame_length=self.cfg.trim_fft_size, hop_length=self.cfg.trim_hop_size)[0]

    def _stft(self, y):
        return librosa.stft(y=y, n_fft=self.cfg.n_fft, hop_length=self.cfg.hop_length, win_length=self.cfg.win_length)

    def _fft(self, y):
        return fft(x=y)

    def _istft(self, y):
        return librosa.istft(y, hop_length=self.cfg.hop_length, win_length=self.cfg.win_length)

    def _amp_to_db(self, x):
        min_level = np.exp(self.cfg.min_db / 20 * np.log(10))
        return 20 * np.log10(np.maximum(min_level, x))

    def _db_to_amp(self, x):
        return np.power(10.0, x * 0.05)

    def _liner2mel(self, mag):
        mel_basis = librosa.filters.mel(self.cfg.sample_rate, self.cfg.n_fft, self.cfg.n_mels)  # (n_mels, 1+n_fft//2)
        mel = np.dot(mel_basis, mag)  # (n_mels, t)
        return mel

    def _normalize(self, S):  # to [0, 1]
        return np.clip((S - self.cfg.ref_db - self.cfg.min_db) / (-self.cfg.min_db), 0, 1)

    def _denormalize(self, S):
        return (np.clip(S, 0, 1) * (-self.cfg.min_db)) + self.cfg.min_db + self.cfg.ref_db

    def _mel_to_linear_matrix(self):
        m = librosa.filters.mel(self.cfg.sample_rate, self.cfg.n_fft, self.cfg.n_mels)
        m_t = np.transpose(m)
        p = np.matmul(m, m_t)
        d = [1.0 / x if np.abs(x) > 1.0e-8 else x for x in np.sum(p, axis=0)]
        return np.matmul(m_t, np.diag(d))

    def _dc_notch_filter(self, wav):
        # code from speex
        notch_radius = 0.982
        den = notch_radius ** 2 + 0.7 * (1 - notch_radius) ** 2
        b = np.array([1, -2, 1]) * notch_radius
        a = np.array([1, -2 * notch_radius, den])
        return signal.lfilter(b, a, wav)

    def _lower_to_uper(self, txt):
        for s in txt:
            if s.islower():
                s = s.uper()

    # print('Ａ', 'Ａ'.upper(), 'Ａ'.lower(), 'Ａ'.lower().upper(), 'A', 'a')
    # global SS
    # SS=[]

    def _txt2num2(self, dict_label, txt):
        style = Style.TONE3
        txt_yj = lazy_pinyin(txt, style)
        num_list = []
        for txt_i in txt_yj:
            if not txt_i in dict_label:
                for s in txt_i:
                    s = s.upper()
                    if not s in dict_label:
                        # SS.append(s)
                        # print(s, 'changed:\n', txt, '\n', txt.replace(s, '-'))
                        # print(list(set(SS)))
                        s = '-'
                        num_list.append(dict_label[s])
                    else:
                        num_list.append(dict_label[s])
            else:
                num_list.append(dict_label[txt_i])

        if txt[-1] != '。':
            num_list.append(dict_label['。'])
        return num_list

    def _show_wav(self, wavs, names):
        for i, wav in enumerate(wavs):
            plt.subplot(421 + i)
            x1 = np.linspace(0, len(wav) - 1, len(wav))
            plt.plot(x1, wav)
            plt.title(names[i])
        plt.show()

    def griffin_lim(self, spectrogram):
        '''Applies Griffin-Lim's raw.
        '''
        X_best = copy.deepcopy(spectrogram)
        for i in range(50):
            X_t = self._istft(X_best)
            est = librosa.stft(X_t, self.cfg.n_fft, self.cfg.hop_length, win_length=self.cfg.win_length)
            phase = est / np.maximum(1e-8, np.abs(est))
            X_best = spectrogram * phase
        X_t = self._istft(X_best)
        y = np.real(X_t)

        return y

    def show_wav_details(self, fpath):
        # Loading sound file
        wavs = []
        names = []
        wav = self.load_wav(fpath)
        wavs.append(wav)
        names.append('raw wav')

        # Trimming
        wav = self.trim_silence(wav)
        wavs.append(wav)
        names.append('Trimming')

        # Preemphasis
        wav_em = self.preemphasize(wav)
        wavs.append(wav_em)
        names.append('Preemphasis')

        # stft
        fft_y = self._fft(wav)
        wavs.append(fft_y)
        names.append('fft')

        # magnitude spectrogram
        mag = np.abs(fft_y)  # (1+n_fft//2, T)
        wavs.append(mag)
        names.append('magnitude spectrogram(abs(fft))')

        normalization_y = mag / len(mag)  # 归一化处理（双边频谱）
        wavs.append(normalization_y)
        names.append('normalization(mag/N)')

        angle_y = np.angle(fft_y)
        wavs.append(angle_y)
        names.append('angel spectrogram')

        self._show_wav(wavs, names)

    def mel_spectrogram(self, fpath):
        '''Returns normalized log(melspectrogram) and log(magnitude) from `sound_file`.
        Args:
          sound_file: A string. The full path of a sound file.

        Returns:
          mel: A 2d array of shape (T, n_mels) <- Transposed
          mag: A 2d array of shape (T, 1+n_fft/2) <- Transposed
     '''
        # Loading sound file
        wav = self.load_wav(fpath)
        # Trimming
        wav = self.trim_silence(wav)
        # Preemphasis
        wav = self.preemphasize(wav)
        # stft
        linear = self._stft(wav)
        # magnitude spectrogram
        mag = np.abs(linear)  # (1+n_fft//2, T)
        # mel spectrogram
        mel = self._liner2mel(mag)
        # to decibel
        mel = self._amp_to_db(mel)
        # normalize
        mel = self._normalize(mel)
        # mel = np.clip((mel - self.cfg.ref_db + self.cfg.max_db) / self.cfg.max_db, 1e-8, 1)
        return mel  # , mag

    def inv_mel_spectrogram(self, mel):
        '''# Generate wave file from spectrogram'''
        # transpose
        # de-noramlize
        mel = self._denormalize(mel)
        # mel = (np.clip(mel, 0, 1) * self.cfg.max_db) - self.cfg.max_db + self.cfg.ref_db
        # to amplitude
        mel = self._db_to_amp(mel)
        m = self._mel_to_linear_matrix()
        mag = np.dot(m, mel)
        # wav reconstruction
        wav = self.griffin_lim(mag)
        # de-preemphasis
        wav = self.de_emphasize(wav)
        # trim
        wav = self.trim_silence(wav)
        return wav.astype(np.float32)

    def text_to_sequence(self, text):
        text = text.strip().replace('“', '').replace('”', '').replace(',', '，').replace(' ', '-')
        text2 = self._txt2num2(self.label_dict, text)
        text2.append(self.label_dict['~'])
        return text2


if __name__ == '__main__':
    txt_path = '/media/lg/SSD_WorkSpace/LG/GitHub/lg_pro_sets/datasets/TTS/ALL_XF/ALL_XF_train.txt'

    wav_path = '/media/lg/DataSet_E/datasets/BZNSYP_16K/waves/000001.wav'
    wav_path = '/media/lg/SSD_WorkSpace/LG/TTS/waveglow_nvidia/output/wav/000001.wav_synthesis.wav'
    from lgdet.util.util_audio import cfg

    #
    _mel_basis = None

    audio = Audio(cfg, test=True)
    # audio.show_wav_details(wav_path)
    # mel = audio.mel_spectrogram(wav_path)
    # torch.save(mel, './mel.mel')
    # voice1 = ua.inv_spectrogram1(mel)
    # wav = audio.inv_mel_spectrogram(mel)
    # ua.write_wav(voice1, 'voice1.wav')
    # audio.write_wav(wav, '/media/lg/SSD_WorkSpace/LG/GitHub/lg_pro_sets/output/voice2.wav')
    import tqdm
    tqd = tqdm.tqdm(open(txt_path).readlines())
    for i, line in enumerate(tqd):
        tmp = line.strip().split('┣┫')
        audio.text_to_sequence(tmp[-1])
    print(list(set(SS)))
