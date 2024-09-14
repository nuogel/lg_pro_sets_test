import torch
import torch.nn as nn

from ..registry import MODELS


@MODELS.registry()
class RNN(nn.Module):
    def __init__(self, cfg):
        super(RNN, self).__init__()

        self.layer_1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=4, padding=1)
        self.layer_2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=4, padding=1)
        self.embedding = nn.Embedding(cfg.TRAIN.CLASS_LENGTH, 10)  # 定义词向量vocab_size个单词，每个单词embed_dim维的词向量

        self.layer_rnn = nn.RNN(
            input_size=cfg.TRAIN.INPUT_SIZE,
            hidden_size=cfg.TRAIN.HIDEN_SIZE,
            num_layers=1,
            batch_first=True)

    def forward(self, train_data):
        x = train_data[0]
        # x [N, 1600, 200]
        net = x.unsqueeze(1)
        net = self.layer_1(net)
        net = self.layer_2(net)
        b, t, f, c = net.size()
        net = net.view((b, t, f * c))
        net, hn = self.layer_rnn(net)  # [2, 64 ,1424]
        # net = torch.nn.functional.softmax(net, dim=2)
        return net


