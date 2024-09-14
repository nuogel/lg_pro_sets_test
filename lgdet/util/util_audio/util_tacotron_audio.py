import librosa
import torch
from lgdet.util.util_audio.common import STFT, dynamic_range_compression, dynamic_range_decompression
import numpy as np
from scipy import signal


class TacotronSTFT(torch.nn.Module):
    def __init__(self, cfg):
        super(TacotronSTFT, self).__init__()
        self.n_mels = cfg.n_mels
        self.sample_rate = cfg.sample_rate
        self.stft_fn = STFT(cfg.n_fft, cfg.hop_length, cfg.win_length)
        mel_basis = librosa.filters.mel(cfg.sample_rate, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

        inv_mel_basis = np.linalg.pinv(mel_basis)
        inv_mel_basis = torch.from_numpy(inv_mel_basis).float()
        self.register_buffer('inv_mel_basis', inv_mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert(torch.min(y.data) >= -1)
        assert(torch.max(y.data) <= 1)

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output

    def inv_mel_spectrogram(self, mel):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mels, T)
        """
        mel = self.spectral_de_normalize(mel.float())
        magnitudes = torch.matmul(self.inv_mel_basis, mel.data)
        magnitudes = torch.max(magnitudes.clone().detach().fill_(1e-10), magnitudes)
        return magnitudes.data

    def griffin_lim(self, magnitudes, n_iters=50, power=2):
        """
        PARAMS
        ------
        magnitudes: spectrogram magnitudes
        stft_fn: STFT class with transform (STFT) and inverse (ISTFT) methods
        """
        magnitudes = magnitudes.unsqueeze(0) ** power
        angles = np.angle(np.exp(2j * np.pi * np.random.rand(*magnitudes.size())))
        angles = angles.astype(np.float32)
        angles = torch.autograd.Variable(torch.from_numpy(angles))
        signal = self.stft_fn.inverse(magnitudes, angles).squeeze(1)

        for i in range(n_iters):
            _, angles = self.stft_fn.transform(signal)
            signal = self.stft_fn.inverse(magnitudes, angles).squeeze(1)
        signal = signal.squeeze()
        # wav = self.de_emphasize(signal)
        return signal

    def de_emphasize(self, wav, preemphasis=0.97):
        return signal.lfilter([1], [1, -preemphasis], wav)

    def in_mel_to_wav(self, mel):
        mag = self.inv_mel_spectrogram(mel)
        wav = self.griffin_lim(mag, n_iters=50)
        return wav


if __name__ == '__main__':
    from lgdet.util.util_audio import cfg
    from lgdet.util.util_audio.util_audio import Audio
    from glob import glob
    import sounddevice as sd
    wav_file = '/media/lg/DataSet_E/datasets/tts/kedaTTS1000_zhiling/wavs/0'
    # wav_path = '/home/lg/datasets/BZNSYP_kedaTTS/waves/0_000001.wav'
    file_list = glob(wav_file + '/*')
    _mel_basis = None
    audio = Audio(cfg, test=True)
    stft = TacotronSTFT(cfg)
    for wav_path in file_list:
        wav_path = '/media/lg/DataSet_E/datasets/BZNSYP/waves/000001.wav'
        wav_raw = audio.load_wav(wav_path)
        wav_trim = audio.trim_silence(wav_raw)
        wav_emp = audio.preemphasize(wav_trim)
        wav_emp_tensor = torch.FloatTensor(wav_emp.astype(np.float32))
        mel = stft.mel_spectrogram(wav_emp_tensor.unsqueeze(0))
        mel = mel.squeeze(0)
        voice2 = stft.in_mel_to_wav(mel)
        # 开始播放
        play_obj = sd.play(audio, 16000)

        # 等待播放结束后退出
        play_obj.wait_done()
        # name = os.path.basename(wav_path)
        # audio.write_wav(voice2, os.path.join(os.path.dirname(wav_path), '../GL', name))
        break
