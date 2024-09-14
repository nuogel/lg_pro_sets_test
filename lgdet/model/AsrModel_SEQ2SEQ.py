import random

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

'''
->不加attention 为84%， 加attention 为82%，why? 尝试修改lg_attention.
->以上是在训练集上为100%，但是在测试集上为0？？？？
->中途遇到list[]长度导致内存泄露；
->改为Lg_attention 后，训练速度慢，并且效果不好。查看是什么原因。

->  单向RNN对训练集效果好，但测试集不得，改为：bidirectional = True；num_layers = 2后，测试集效果不错。
作用no_att :[WRR]: 0.7882364684852814  [SRR]: 0.79453125  [CRR]: 0.8356708720119862
使用raw_att:[WRR]: 0.23902110332813387  [SRR]: 0.28463855421686746  [CRR]: 0.4100081112949657
使用lg_att: [不收敛！！！]
'''

from ..registry import MODELS


@MODELS.registry()
class SEQ2SEQ(nn.Module):
    def __init__(self, cfg):
        super(SEQ2SEQ, self).__init__()
        self.cfg = cfg
        input_size = cfg.TRAIN.AUDIO_FEATURE_LENGTH
        vocab_size = cfg.TRAIN.CLASS_LENGTH
        hidden_size = 1024  # 也用于embedding,越大loss下降越快
        num_layers = 2
        dropout = 0.2
        bidirectional = True
        sample_rate = .3
        self.encoder = Encoder(input_size, hidden_size, num_layers, dropout, bidirectional)  # 一个GRU
        self.decoder = Decoder(vocab_size, hidden_size, sample_rate, self.cfg)
        self.vocab_size = vocab_size
        self.add_attention = None  # 'raw_att'  #'lg_att'

    def forward(self, **args):
        '''
        `inputs`: (batch, length, dim)
        `targets`: (batch, length)
        '''
        inputs, targets, _, __ = args['input_data']  # INPUT:[B, T, H]
        enc_y, c_hid = self.encoder(inputs)  # 一个GRU，获得输出层和隐藏层[B, T, H], [B,H]，c_hid 相当于C.
        if args['is_training']:
            out = self.decoder(targets, enc_y, c_hid, self.add_attention)
        else:
            out, logp = self.greedy_decode(enc_y, c_hid, self.add_attention)
            # out, logp = self.beam_search(enc_y)
        return out

    def greedy_decode(self, enc_y, c_hid, add_attention):
        """ upport batch sequences """
        batch_size = enc_y.size(0)
        start_number = self.cfg.TRAIN.CLASS_LENGTH - 2  # 1209==START
        inputs = torch.LongTensor([start_number] * batch_size)  # 随便给一个启动sequence, 实验表明，我给任意数都行！。？？最后还是加了起始和结束标记
        inputs = inputs.to(self.cfg.TRAIN.DEVICE)
        y_seqs = []
        STOP = False  # give a number not equal to 0
        ax = sx = None
        while STOP is False:  # 0 represent the last word [end].
            output, c_hid, ax, sx = self.decoder._step(inputs, c_hid, enc_y, ax, sx, add_attention)
            output = torch.softmax(output, dim=-1)
            score, inputs = output.max(dim=-1)  # 将上一次预测的结果作为下一次的输入
            labels = [int(_inputs.item()) for _inputs in inputs]
            y_seqs.append(labels)
            if sum(labels) == 0 or (len(y_seqs) > 70):
                STOP = True
        y_seqs = np.array(y_seqs)
        y_seqs = np.transpose(y_seqs, (1, 0))
        return y_seqs, 0

    # def beam_search(self, xs, beam_size=10, max_len=200):
    #     def decode_step(self, x, y, state=None, softmax=False):
    #         """ `x` (TH), `y` (1) """
    #         if state is None:
    #             hx, ax, sx = None, None, None
    #         else:
    #             hx, ax, sx = state
    #         out, hx, ax, sx = self.decoder._step(y, hx, x, ax, sx)
    #         if softmax:
    #             out = nn.functional.log_softmax(out, dim=1)
    #         return out, (hx, ax, sx)
    #
    #     start_tok = self.vocab_size - 1;
    #     end_tok = 0
    #     x, h = self.encode(xs)
    #     y = torch.autograd.Variable(torch.LongTensor([start_tok]), volatile=True)
    #     beam = [((start_tok,), 0, (h, None, None))]
    #     complete = []
    #     for _ in range(max_len):
    #         new_beam = []
    #         for hyp, score, state in beam:
    #             y[0] = hyp[-1]
    #             out, state = decode_step(x, y, state=state, softmax=True)
    #             out = out.cpu().data.numpy().squeeze(axis=0).tolist()
    #             for i, p in enumerate(out):
    #                 new_score = score + p
    #                 new_hyp = hyp + (i,)
    #                 new_beam.append((new_hyp, new_score, state))
    #         new_beam = sorted(new_beam, key=lambda x: x[1], reverse=True)
    #
    #         # Remove complete hypotheses
    #         for cand in new_beam[:beam_size]:
    #             if cand[0][-1] == end_tok:
    #                 complete.append(cand)
    #
    #         beam = filter(lambda x: x[0][-1] != end_tok, new_beam)
    #         beam = beam[:beam_size]
    #
    #         if len(beam) == 0:
    #             break
    #
    #         # Stopping criteria:
    #         # complete contains beam_size more probable
    #         # candidates than anything left in the beam
    #         if sum(c[1] > beam[0][1] for c in complete) >= beam_size:
    #             break
    #
    #     complete = sorted(complete, key=lambda x: x[1], reverse=True)
    #     if len(complete) == 0:
    #         complete = beam
    #     hyp, score, _ = complete[0]
    #     return hyp, score


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)

    def forward(self, inputs, hidden=None):
        '''
        `inputs`: (batch, length, input_size)
        `hidden`: Initial hidden state (num_layer, batch_size, hidden_size)
        '''
        x, h = self.rnn(inputs, hidden)
        dim = h.shape[0]
        h = h.sum(dim=0) / dim
        if self.rnn.bidirectional:  # WHY?
            half = x.shape[-1] // 2
            x = x[:, :, :half] + x[:, :, half:]
        return x, h


class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, sample_rate, cfg, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.attention = NNAttention(hidden_size, log_t=True)
        self.lg_attention = LGAttention(hidden_size, log_t=True)

        self.rnn = nn.GRUCell(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)  # no 'sos'
        self.vocab_size = vocab_size
        self.sample_rate = sample_rate
        self.cfg = cfg

    def forward(self, target, enc_y, c_hid, add_attention):  # targets, enc_y, c_hid
        '''
        `target`: (batch, length)
        `enc_y`: Encoder output, (batch, length, dim)
        `c_hid`: last hidden state of encoder
        '''
        target = target.transpose(0, 1)
        length = target.shape[0]
        batch_size = enc_y.size(0)
        # <START>
        start_number = self.cfg.TRAIN.CLASS_LENGTH - 2  # 1209==START
        target_i = torch.LongTensor([start_number] * batch_size).to(enc_y.device)  # 随便给一个启动sequence, 实验表明，我给任意数都行！。？？最后还是加了起始和结束标记

        ax = sx = None
        out = []
        align = []
        pre = torch.zeros((self.cfg.TRAIN.BATCH_SIZE, length, self.cfg.TRAIN.CLASS_LENGTH)).to(self.cfg.TRAIN.DEVICE)
        for i in range(length):
            output, c_hid, ax, sx = self._step(target_i, c_hid, enc_y, ax, sx, add_attention)
            if random.random() < self.sample_rate:
                target_i = output.max(dim=1)[1]
            else:
                target_i = target[i]
            pre[:, i, :] = output
            out.append(output)
            align.append(ax)
        # out = torch.cat(out, dim=0)
        # out = out.view(-1, out.shape[-1])

        return pre

    def _step(self, target_i, c_hid, enc_y, ax, sx, add_attention='raw_att'):
        embeded = self.embedding(target_i)
        if add_attention == 'lg_att':
            at = self.lg_attention(enc_y, c_hid, embeded)
            gru_cell_y = self.rnn(c_hid, at)
            output = self.fc(gru_cell_y)
        else:
            if sx is not None:
                # last context vector
                embeded = embeded + sx
            gru_cell_y = self.rnn(embeded, c_hid)
            if add_attention == 'raw_att':
                sx, ax = self.attention(enc_y, gru_cell_y, ax)
                output = self.fc(gru_cell_y + sx)
            else:
                output = self.fc(gru_cell_y)
        return output, gru_cell_y, ax, sx


class NNAttention(nn.Module):

    def __init__(self, n_channels, kernel_size=15, log_t=False):
        super(NNAttention, self).__init__()
        assert kernel_size % 2 == 1, \
            "Kernel size should be odd for 'same' conv."
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(1, n_channels, kernel_size, padding=padding)
        self.nn = nn.Sequential(
            nn.ReLU(),
            nn.Linear(n_channels, 1))
        self.log_t = log_t

    def forward(self, enc_y, gru_cell_y, ax=None):
        """ `enc_y` (BTH), `gru_cell_y` (BH) """
        pax = enc_y + gru_cell_y.unsqueeze(dim=1)  # BTH

        if ax is not None:
            ax = ax.unsqueeze(dim=1)  # B1T
            ax = self.conv(ax).transpose(1, 2)  # BTH
            pax = pax + ax

        pax = self.nn(pax)  # BT1  #线性变换，形成单字的attention
        pax = pax.squeeze(dim=2)
        if self.log_t:
            log_t = math.log(pax.shape[1])
            pax = log_t * pax
        ax = nn.functional.softmax(pax, dim=1)  # BT

        sx = ax.unsqueeze(2)  # BT1
        sx = torch.sum(enc_y * sx, dim=1)  # BH
        return sx, ax


class LGAttention(nn.Module):
    def __init__(self, n_channels, kernel_size=15, log_t=False):
        super(LGAttention, self).__init__()
        assert kernel_size % 2 == 1, \
            "Kernel size should be odd for 'same' conv."
        self.nn = nn.Sequential(
            nn.Linear(n_channels * 2, n_channels),
            nn.ReLU())
        self.log_t = log_t

    def forward(self, enc_y, c_hid, embeded):
        """ `enc_y` (BTH), `c_hid` (BH)
         reference: https://guillaumegenthial.github.io/sequence-to-sequence.html
         """
        _c_hid = c_hid.unsqueeze(dim=1)
        at = enc_y * _c_hid
        a_hat = at.softmax(1)
        ct = a_hat * enc_y
        ct = torch.sum(ct, dim=1)
        ct = torch.cat((embeded, ct), dim=1)
        ct = self.nn(ct)
        return ct
