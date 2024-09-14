import librosa
import librosa.filters
import math
import numpy as np
from scipy import signal
from scipy.io import wavfile
import lws
from nnmnkwii.preprocessing import preemphasis, inv_preemphasis


class Audio:
    def __init__(self, cfg):
        self.cfg = cfg

    def read_wav(self, path):
        return librosa.core.load(path, sr=self.cfg.sample_rate)[0]

    def write_wav(self, wav, path):
        wav = wav * 32767 / max(0.01, np.max(np.abs(wav)))
        wavfile.write(path, self.cfg.sample_rate, wav.astype(np.int16))

    def melspectrogram(self, y):
        D = self._lws_processor().stft(self._preemphasis(y)).T
        S = self._amp_to_db(self._linear_to_mel(np.abs(D))) - self.cfg.ref_db
        if not self.cfg.allow_clipping_in_normalization:
            assert S.max() <= 0 and S.min() - self.cfg.min_db >= 0
        return self._normalize(S)

    def inv_spectrogram1(self, spectrogram):
        '''Converts spectrogram to waveform using librosa'''
        S = self._db_to_amp(self._denormalize(spectrogram) + self.cfg.ref_db)  # Convert back to linear
        return self._inv_preemphasis(self._griffin_lim(S ** self.cfg.power))  # Reconstruct phase

    def inv_spectrogram2(self, spectrogram):
        '''Converts spectrogram to waveform using librosa'''
        S = self._db_to_amp(self._denormalize(spectrogram) + self.cfg.ref_db)  # Convert back to linear
        processor = self._lws_processor()
        D = processor.run_lws(S.astype(np.float64).T ** self.cfg.power)
        y = processor.istft(D).astype(np.float32)
        return self._inv_preemphasis(y)

    def _preemphasis(self, x):
        return preemphasis(x, self.cfg.preemphasis)

    def _inv_preemphasis(self, x):
        return inv_preemphasis(x, self.cfg.preemphasis)

    def _griffin_lim(self, S):
        '''librosa implementation of Griffin-Lim
        Based on https://github.com/librosa/librosa/issues/434
        '''
        angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
        S_complex = np.abs(S).astype(np.complex)
        y = self._istft(S_complex * angles)
        for i in range(60):
            angles = np.exp(1j * np.angle(self._stft(y)))
            y = self._istft(S_complex * angles)
        return y

    def _istft(self, y):
        # n_fft, hop_length, win_length = self._stft_parameters()
        return librosa.istft(y, hop_length=self.cfg.hop_length, win_length=self.cfg.win_length)

    def _stft(self, y):
        # n_fft, hop_length, win_length = self._stft_parameters()
        return librosa.stft(y=y, n_fft=self.cfg.n_fft, hop_length=self.cfg.hop_length, win_length=self.cfg.win_length)

    def _stft_parameters(self):
        n_fft = (self.cfg.num_freq - 1) * 2
        hop_length = int(self.cfg.frame_shift_ms / 1000 * self.cfg.sample_rate)
        win_length = int(self.cfg.frame_length_ms / 1000 * self.cfg.sample_rate)
        return n_fft, hop_length, win_length

    def _lws_processor(self, ):
        return lws.lws(self.cfg.n_fft, self.cfg.hop_length, mode="speech")

    # Conversions:

    def _linear_to_mel(self, spectrogram):
        global _mel_basis
        if _mel_basis is None:
            _mel_basis = self._build_mel_basis()
        return np.dot(_mel_basis, spectrogram)

    def _build_mel_basis(self, ):
        return librosa.filters.mel(self.cfg.sample_rate, self.cfg.n_fft, n_mels=self.cfg.n_mels)

    def _amp_to_db(self, x):
        min = np.exp(self.cfg.min_db / 20 * np.log(10))
        return 20 * np.log10(np.maximum(min, x))

    def _db_to_amp(self, x):
        return np.power(10.0, x * 0.05)

    def _normalize(self, S):
        return np.clip((S - self.cfg.min_db) / -self.cfg.min_db, 0, 1)

    def _denormalize(self, S):
        return (np.clip(S, 0, 1) * -self.cfg.min_db) + self.cfg.min_db


if __name__ == '__main__':
    wav_path = '/home/lg/datasets/BZNSYP_kedaTTS/waves/0_000001.wav'
    from lgdet.util.util_voice import cfg

    _mel_basis = None

    ua = Audio(cfg)
    wav = ua.read_wav(wav_path)
    mel = ua.melspectrogram(wav)
    # voice1 = ua.inv_spectrogram1(mel)
    voice2 = ua.inv_spectrogram2(mel)
    # ua.write_wav(voice1, 'voice1.wav')
    ua.write_wav(voice2, 'voice2.wav')
