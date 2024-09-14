import torch
import torch.nn as nn
from lgdet.util.util_audio.util_audio import Audio
from lgdet.util.util_audio.util_tacotron_audio import TacotronSTFT
import numpy as np


class TACOTRONLOSS:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.TRAIN.DEVICE
        self.mseloss = torch.nn.MSELoss()
        self.bcelogistloss = torch.nn.BCEWithLogitsLoss()
        self.audio = Audio(cfg)
        self.stft = TacotronSTFT(cfg.TRAIN)

    def Loss_Call(self, predicted, train_data, kwargs):
        global_step = kwargs['global_step']
        mel_out_before, mel_out_after, gate_out, _ = predicted
        mel_target, gate_target = train_data[1]
        mel_loss = self.mseloss(mel_out_before, mel_target) + self.mseloss(mel_out_after, mel_target)
        gate_loss = self.bcelogistloss(gate_out.view(-1, 1), gate_target.view(-1, 1))
        # mel_loss = nn.MSELoss()(mel_out_before, mel_target) + nn.MSELoss()(mel_out_after, mel_target)
        # gate_loss = nn.BCEWithLogitsLoss()(gate_out.view(-1, 1), gate_target.view(-1, 1))
        total_loss = mel_loss + gate_loss
        metrics = {'mel_loss': mel_loss,
                   'gate_loss': gate_loss}
        if (global_step) % 10000 == 0:
            length = train_data[0][3][0]
            txt = train_data[3][0][1]
            mel_out_after = mel_out_after.cpu()[0][..., :length]
            mel_target = mel_target.cpu()[0][..., :length]
            waveform = self.stft.in_mel_to_wav(mel_target)
            self.audio.write_wav(waveform, 'output/train/%s_gt.wav' % (txt))
            waveform = self.stft.in_mel_to_wav(mel_out_after)
            self.audio.write_wav(waveform, 'output/train/%s_pre.wav' % (txt))
        return {'total_loss': total_loss, 'metrics': metrics}
