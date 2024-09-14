import os
import torch
import numpy as np
from .loader_base import BaseLoader
from lgdet.util.util_audio.util_audio import Audio
from lgdet.util.util_audio.util_tacotron_audio import TacotronSTFT
from ..registry import DATALOADERS


@DATALOADERS.registry()
class TTS_Loader(BaseLoader):
    def __init__(self, cfg, dataset, is_training):
        super(TTS_Loader, self).__init__(cfg, dataset, is_training)
        self.audio = Audio(cfg)
        self.stft = TacotronSTFT(cfg.TRAIN)

    def __getitem__(self, index):
        if self.one_test:
            data_info = self.dataset_infos[0]
        else:
            data_info = self.dataset_infos[index]
        [wav_path, txt] = data_info
        if self.is_training:
            audio_norm = self.audio.load_wav(os.path.join(self.datapath, wav_path))
            # wav_trim = self.audio.trim_silence(wav_raw)
            # wav_emp = self.audio.preemphasize(wav_trim)
            # wav_emp_tensor = torch.FloatTensor(wav_emp.astype(np.float32))
            # mel = self.stft.mel_spectrogram(wav_emp_tensor.unsqueeze(0))
            # mel = mel.squeeze(0)

            audio_norm = torch.FloatTensor(audio_norm.astype(np.float32))
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            mel = self.stft.mel_spectrogram(audio_norm)
            mel = torch.squeeze(mel, 0)
        else:
            mel = None
        txt_seq = self.audio.text_to_sequence(txt)
        len_seq = len(txt_seq)
        group = (txt_seq, mel, len_seq, data_info)  # for i in range(speakers)
        return group

    def collate_fun(self, batch):
        """Create batch"""
        # Right zero-pad all one-hot text sequences to max input length
        seq_lens, ids_sorted_decreasing = torch.sort(
            torch.IntTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_seq_len = seq_lens[0]

        seqs = []
        for i in range(len(ids_sorted_decreasing)):
            seq = batch[ids_sorted_decreasing[i]][0]
            seqs.append(np.pad(seq, [0, max_seq_len - len(seq)], mode='constant'))

        if self.is_training:
            # Right zero-pad mel-spec
            max_target_len = max([x[1].shape[1] for x in batch])
            if max_target_len % self.cfg.TRAIN.n_frames_per_step != 0:
                max_target_len += self.cfg.TRAIN.n_frames_per_step - max_target_len % self.cfg.TRAIN.n_frames_per_step
                assert max_target_len % self.cfg.TRAIN.n_frames_per_step == 0

            # include mel padded and gate padded
            targets, reduced_targets = [], []
            gates = np.zeros([len(batch), max_target_len], dtype=np.float32)
            target_lengths = torch.IntTensor(len(batch))
            for i in range(len(ids_sorted_decreasing)):
                mel = batch[ids_sorted_decreasing[i]][1]
                target_lengths[i] = mel.shape[1]
                gates[i, mel.shape[1] - 1:] = 1
                padded_mel = np.pad(mel, [(0, 0), (0, max_target_len - mel.shape[1])], mode='constant', constant_values=-4)
                targets.append(padded_mel)
                reduced_mel = padded_mel[:, ::self.cfg.TRAIN.n_frames_per_step]
                reduced_targets.append(reduced_mel)
            targets = torch.from_numpy(np.stack(targets))
            reduced_targets = torch.from_numpy(np.stack(reduced_targets))
            gates = torch.from_numpy(gates)
            num_frames = torch.sum(target_lengths)
        else:
            reduced_targets = None
            target_lengths = None
            targets = None
            gates = None
            num_frames = None

        seqs = torch.from_numpy(np.stack(seqs))
        x = [seqs, seq_lens, reduced_targets, target_lengths]
        y = [targets, gates]

        data_infos = []
        for i in range(len(ids_sorted_decreasing)):
            info = batch[ids_sorted_decreasing[i]][3]
            data_infos.append(info)
        return x, y, num_frames, data_infos
