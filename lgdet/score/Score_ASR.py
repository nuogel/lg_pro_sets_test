from lgdet.dataloader.Loader_ASR import DataLoader
import Levenshtein as Lev
import numpy as np


class Score:  # WER（字错误率）/CER（字符错误率）和SER（句错误率）
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataloader = DataLoader(cfg)

    def init_parameters(self):
        self.rate_all = np.asarray([0., 0., 0.], np.float32)
        self.rate_batch = np.asarray([0., 0., 0.], np.float32)
        self.batches = 0

    def cal_score(self, pre, gt_data):  # SER
        self.rate_batch = np.asarray([0., 0., 0.], np.float32)
        _, target, input_lengths, target_lengths = gt_data
        # print('batch NO:', self.batches)
        for i in range(self.cfg.TRAIN.BATCH_SIZE):
            pre_i = pre[i][pre[i] > 0]
            gt_i = target[i, :target_lengths[i]-1].cpu()
            s_pr = " ".join(str(i) for i in self.dataloader._number2pinying(pre_i))
            s_gt = " ".join(str(i) for i in self.dataloader._number2pinying(gt_i.numpy().tolist()))  # [1:-1]

            # if len(pre_i) != target_lengths[i]: continue
            # pre_i = torch.Tensor(pre_i)
            # yes = torch.eq(pre_i[:target_lengths[i]], gt_i.type(torch.FloatTensor))
            # self.rate_batch += float(yes.sum() / target_lengths[i])
            WER = self._wer(s_pr, s_gt) / len(s_gt.split())
            CER = self._cer(s_pr, s_gt) / len(s_gt.replace(' ', ''))
            SER = 1 if WER != 0. or CER != 0. else 0.

            self.rate_batch[0] += WER
            self.rate_batch[1] += SER
            self.rate_batch[2] += CER
            if self.cfg.TEST.SHOW_PREDICTED:
                print('PR: ', s_pr)
                print('GT: ', s_gt)
                print('[WER]:', WER, ' [SER]:', SER, ' [CER]:', CER)
        self.rate_all += self.rate_batch / self.cfg.TRAIN.BATCH_SIZE
        self.batches += 1

    def score_out(self):
        rate = 1. - self.rate_all / self.batches
        print('[WRR]:', rate[0], ' [SRR]:', rate[1], ' [CRR]:', rate[2])
        return rate[0], rate[1], rate[2]

    def _wer(self, s1, s2):
        """
        Computes the Word Error Rate, defined as the edit distance between the
        two provided sentences after tokenizing to words.
        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """

        # build mapping of words to integers
        b = set(s1.split() + s2.split())
        word2char = dict(zip(b, range(len(b))))

        # map the words to a char array (Levenshtein packages only accepts
        # strings)
        w1 = [chr(word2char[w]) for w in s1.split()]
        w2 = [chr(word2char[w]) for w in s2.split()]

        return Lev.distance(''.join(w1), ''.join(w2))

    def _cer(self, s1, s2):
        """
        Computes the Character Error Rate, defined as the edit distance.

        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """
        s1, s2, = s1.replace(' ', ''), s2.replace(' ', '')
        return Lev.distance(s1, s2)
