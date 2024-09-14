import torch
from torch.nn import CTCLoss
from lgdet.util.util_to_onehot import to_onehot
from lgdet.dataloader.Loader_ASR import ASR_Loader


class RnnLoss:
    def __init__(self, cfg):
        self.cfg = cfg
        self.loss_ctc = CTCLoss(reduction='sum', blank=len('CHARS')-1)
        self.loss_mse = torch.nn.MSELoss(reduction='sum')
        self.data = ASR_Loader(cfg)

    def Loss_Call(self, input, train_data, losstype='mse'):
        _, target, input_lengths, target_lengths = train_data
        if losstype == 'ctc':
            batch_size = input.shape[1]
            input_lengths = input_lengths // 8
            loss = self.loss_ctc(input, target, input_lengths, target_lengths) / batch_size, None

        elif losstype == 'mse':
            loss = self._mse_loss(input, target, input_lengths, target_lengths)
        else:
            loss = -1
        return loss

    def _mse_loss(self, input, target, input_lengths=None, target_lengths=None):
        target = to_onehot(target, self.cfg.TRAIN.CLASS_LENGTH).to(self.cfg.TRAIN.DEVICE)
        # print(target[0, 0])
        loss = self.loss_mse(input, target)
        precision = 0
        N = input.size()[0]
        for i in range(N):
            target_lengths_i = target_lengths[i]
            lab_py_id = torch.argmax(target[i][:target_lengths_i], -1)
            lab_py = self.data._number2pinying(lab_py_id)
            print("the lab_pingyin is :", lab_py)
            pre_py_id = torch.argmax(input[i][:target_lengths_i], -1)
            pre_py = self.data._number2pinying(pre_py_id)
            print("the pingyin is :", pre_py)
            precision += float(sum(lab_py_id.eq(pre_py_id))) / len(pre_py)
        return loss / N, precision / N
