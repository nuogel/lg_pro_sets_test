import torch
import torch.nn as nn
from collections import OrderedDict
from lgdet.util.util_ctc_decoder import beam_decode
from keras import backend as K
import numpy as np

'''
-》只用CNN，若去掉 layers.append(('batchnorm', nn.BatchNorm2d(out_ch))) 梯度很容易爆炸；加上后收敛很快。
-》在loss function 后除以batch后，梯度爆炸消失；
-》若去掉 layers.append(('batchnorm', nn.BatchNorm2d(out_ch)))，使用RNN，LSTM ,GRU 梯度消失；加上后一样，原因：
-》RNN中采用nonlinearity='relu'时，当loss=8左右时，出现nan; 去掉不出现，但收敛很慢，可能梯度消失。
-》当使用cnn时，也会出现梯度消失和爆炸；猜想是损失函数处出了问题。
->小批量收敛，大批量不收敛；
->加入nn.Hardtanh(-50, 50）控制输出后，训练变得正常，好像还存在loss=220左右时，梯度很小的情况，不知在linear层加入batchnorm 会如何。
->加入nn.BatchNorm1d后，训练变得稳定，不知梯度情况如何， 依然loss=220左右时，梯度很小的情况，然后出现nan;
->在Conv 后增加x = x.transpose(1, 2)使时间序列排在第2，成为NTCH,而不是直接变形view();没发现的细节，CNN损失下降正常了。GRU出现nan; 
->对于GRU出现nan，在GRU后面加两层fc后情况好转但在220左右下降慢；到210时出现nan; 
->对于GRU出现nan, 在GRU后面加入batchnorm层就好了；当损失到58左右时出现nan ; 
->对于CNN加入batchnorm层，下降速度变慢；直接用两层fc速度快；
->对于GRU，在其后加入x = x.transpose(1, 2)；x = self.batchnorm1d(x)；x = x.transpose(1, 2)后，下降稳定，但其在loss=40左右时，又出现nan; 
->无端点检测的decode_lg为[WRR]: 0.9513495792309238  [SRR]: 0.27320359281437123  [CRR]: 0.9701372321852586
->加入端点检测的：[WRR]: 0.9134984615058939  [SRR]: 0.11953124999999998  [CRR]: 0.948527428791817，不知是不是没训练到位。
'''

from ..registry import MODELS


@MODELS.registry()
class CTC(nn.Module):  # write by LG
    def __init__(self, cfg):
        super(CTC, self).__init__()
        cnn_last_size = 128
        bidirectional = True
        direction = 2 if bidirectional else 1
        liner_mid_size = 1024  # 如果设置太小，很难收敛，收敛慢。中间神经元越多收敛越快。但显存消耗大。
        self.cnn_1 = self.make_layer(1, 32)
        self.cnn_2 = self.make_layer(32, 64)
        self.cnn_3 = self.make_layer(64, 128)
        self.cnn_4 = self.make_layer(128, cnn_last_size, max_pool=0)
        cnn_input_size = cfg.TRAIN.AUDIO_FEATURE_LENGTH // 8 * cnn_last_size  # 200//16*128
        self.linear_0 = nn.Linear(cnn_input_size, liner_mid_size, bias=False)
        self.linear_1 = nn.Linear(liner_mid_size, liner_mid_size, bias=False)
        self.linear_2 = nn.Linear(liner_mid_size, cfg.TRAIN.CLASS_LENGTH, bias=False)
        self.batchnorm1d_0 = nn.BatchNorm1d(liner_mid_size)
        self.batchnorm1d_1 = nn.BatchNorm1d(liner_mid_size)
        self.batchnorm1d_2 = nn.BatchNorm1d(cfg.TRAIN.CLASS_LENGTH)
        self.hardtanh = nn.Hardtanh(-50, 50, inplace=True)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.layer_rnn = nn.RNN(input_size=cnn_input_size, hidden_size=liner_mid_size//direction, num_layers=2, batch_first=True, bidirectional=bidirectional)
        self.layer_lstm = nn.LSTM(input_size=cnn_input_size, hidden_size=liner_mid_size//direction, num_layers=2, batch_first=True, bidirectional=bidirectional)
        self.layer_gru = nn.GRU(input_size=cnn_input_size, hidden_size=liner_mid_size//direction, num_layers=2, batch_first=True, bidirectional=bidirectional)
        self.rnn_Dict = {1: self.layer_rnn, 2: self.layer_lstm, 3: self.layer_gru}
        self.ctc_type = 2  # [0->cnn +fc+ctc , 1-> cnn + rnn +fc+ctc ;2->cnn+lstm+fc+ctc;3->cnn+gru+fc+ctc]

    def make_layer(self, in_ch, out_ch, ksize=3, stride=1, max_pool=2, conv=True, up_samping=0, last_layer=False):
        layers = []
        if conv:
            padding = (ksize - 1) // 2
            layers.append(('conv2d', nn.Conv2d(in_ch, out_ch, kernel_size=ksize, stride=stride, padding=padding)))
            if not last_layer:  # the last layer don't need this activation functions.
                layers.append(('batchnorm', nn.BatchNorm2d(out_ch)))
                layers.append(('leakyrelu', nn.LeakyReLU()))
        if max_pool:
            layers.append(('max_pool', nn.MaxPool2d(max_pool, max_pool)))
        if up_samping:
            layers.append(('up_samping', nn.Upsample(scale_factor=up_samping, mode='bilinear', align_corners=True)))

        return nn.Sequential(OrderedDict(layers))

    def forward(self, **args):
        x = args['input_x']
        x = x.unsqueeze(1)
        x = self.cnn_1(x)
        x = self.cnn_2(x)
        x = self.cnn_3(x)
        x = self.cnn_4(x)
        x = x.transpose(1, 2)
        b, t, c, f = x.size()
        x = x.contiguous().view((b, t, f * c))
        if self.ctc_type == 0:
            x = self.linear_0(x)
            x = self.relu(x)
            x = self.linear_2(x)

        elif self.ctc_type != 0:
            x, _ = self.rnn_Dict[self.ctc_type](x)
            x = x.transpose(1, 2)
            x = self.batchnorm1d_0(x)
            x = x.transpose(1, 2)
            x = self.relu(x)
            x = self.linear_1(x)
            x = self.relu(x)
            x = self.linear_2(x)

        if args['is_training']:
            x = x.transpose(0, 1)
            x = x.log_softmax(dim=-1)  # ctc 输入要求
        else:
            x = x.softmax(dim=-1)
            # x = self.decode_ctc_K(x, train_data[2] // 8, )
            x = self.ctc_greedy_decode_LG(x)
            # x = self.beam_decode_For(x)
        return x

    def beam_decode_For(self, y_pred):  # cpu 版，太慢。
        # print('beam_decode for CTC')
        decode_out = []
        for i in range(y_pred.shape[0]):
            # input = xs[i].cpu().detach().numpy()
            input = y_pred[i]
            out, score = beam_decode(input)
            decode_out.append(out)
        return decode_out

    def ctc_greedy_decode_LG(self, y_pred):  # greedy by LG
        # print('ctc_greedy_decode for CTC')
        logp, pred = torch.max(y_pred, dim=-1)
        pre = pred.data.cpu().numpy()
        pre_list_all = []
        for batch in range(pre.shape[0]):
            pre_list = [0]  # tmp
            for num in pre[batch]:
                if num != 0 and num != pre_list[-1]:  # drop the _ and repeat word.
                    pre_list.append(num)
            pre_list_all.append(np.asarray(pre_list, np.float32))
        return pre_list_all

    def decode_ctc_K(self, y_pred, input_length, greedy=False):  # okay
        # input = torch.transpose(input, 0, 1)
        pre_list_all = []
        y_pred_input = y_pred.cpu().detach().numpy()
        input_length = input_length.cpu().detach().numpy()
        decoded = K.ctc_decode(y_pred_input, input_length, greedy, beam_width=100,
                               top_paths=1)  # greedy decode is okay, beam is okay.
        for batch_i in range(y_pred.shape[0]):
            r_i = K.get_value(decoded[0][0][batch_i])
            pre_list_all.append(r_i)
        return pre_list_all

# n, t = x.size(0), x.size(1)
# x = x.contiguous().view(n*t, -1)
# # x = self.batch_fc_1(x)
# x = self.batch_fc_2(x)
# x = x.contiguous().view(n, t, -1)
