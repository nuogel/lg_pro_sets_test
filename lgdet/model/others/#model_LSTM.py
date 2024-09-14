import torch
import torch.nn as nn
from torch.autograd import Variable
from cfg import config as cfg


class ASR_MODEL(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ASR_MODEL, self).__init__()

        self.layer_1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=4, padding=1)
        self.layer_2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=4, padding=1)
        self.embedding = nn.Embedding(cfg.CLASS_LENGTH, 10)  # 定义词向量vocab_size个单词，每个单词embed_dim维的词向量

        self.fc = nn.Linear(1300, 1424)
        self.layer_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True)

        self.layer_grucell = nn.GRUCell(input_size=input_size,  # 定义一个GRU单元GRUCELL(3000, 1424)
                                        hidden_size=hidden_size)

        self.layer_gru = nn.GRU(input_size=input_size,
                                hidden_size=hidden_size)
        '''
        rnn = nn.GRUCell(10, 20)
        input = Variable(torch.randn(6, 3, 10))
        hx = Variable(torch.randn(3, 20))
        output = []
        for i in range(6):
             hx = rnn(input[i], hx)
             output.append(hx)
        '''


    def forward(self, x):
        # x [N, 1600, 200]
        net = x.unsqueeze(1)
        model = "mse"
        if model == "ctc":
            net = self.layer_1(net)
            net = self.layer_2(net)
            net = torch.transpose(net, 1, 2).contiguous()
            b, t, f, c = net.size()
            net = net.view((b, t, f * c))

            net, (hn, cn) = self.layer_lstm(net)  # [2, 400 ,1424]
            # net, hn = self.layer_gru(net)
            fc=True
            if fc:
                y_pre_size = net.size()
                net = net.contiguous().view(y_pre_size[0], -1)
                net = self.fc(net)
                net = net.view(y_pre_size[0], -1, y_pre_size[-1])
            net = torch.nn.functional.softmax(net, dim=2)
            return net
        if model == "mse":
            net = self.layer_1(net)
            net = self.layer_2(net)
            b, t, f, c = net.size()
            net = net.view((b, t, f * c))

            net, (hn, cn) = self.layer_lstm(net)  # [2, 400 ,1424]


            # net = torch.nn.functional.softmax(net, dim=2)
            return net

    '''
    LSTM:
    inputs = torch.randn(5,3,10)
    有3个句子，每个句子5个单词，每个单词用10维的向量表示
    seq_len=5,bitch_size=3,input_size=10
    
    假设有100个句子（sequence）,每个句子里有7个词，batch_size=64，embedding_size=300
    此时，各个参数为：
    input_size=embedding_size=300
    batch=batch_size=64
    seq_len=7
    
    另外设置hidden_size=100, num_layers=1
    import torch
    import torch.nn as nn
    lstm = nn.LSTM(300, 100, 1)
    x = torch.randn(7, 64, 300)
    h0 = torch.randn(1, 64, 100)
    c0 = torch.randn(1, 64, 100)
    output, (hn, cn)=lstm(x, (h0, c0))
    
    >>
    output.shape  torch.Size([7, 64, 100])
    hn.shape  torch.Size([1, 64, 100])
    cn.shape  torch.Size([1, 64, 100])

    '''

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                # if m.bias is not None:
                #     m.bias.data.zero_()
                weight = m.weight
                torch.nn.init.kaiming_normal_(weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
