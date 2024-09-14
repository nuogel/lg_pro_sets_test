"""Calculate the F1 score."""
from lgdet.registry import SCORES
import torch


@SCORES.registry()
class Score:
    """Calculate the F1 score."""

    def __init__(self, cfg):
        """Init the parameters."""
        self.pre_corr_dict = {}  # store the correct times for cls A
        self.gt_item_dict = {}  # store the GT times for cls A

    def init_parameters(self):
        self.score = 0.
        self.batches = 0

    def cal_score(self, pre_labels, test_data=None, from_net=True):
        gt_labels = test_data[1]
        correct = torch.argmax(torch.softmax(pre_labels, -1), -1) == gt_labels
        gtlist = list(gt_labels.cpu().numpy())
        gts = list(set(gtlist))
        for gti in gts:
            corri = correct[gtlist == gti]
            if gti not in self.pre_corr_dict:
                self.pre_corr_dict[gti] = sum(corri).item()
                self.gt_item_dict[gti] = gtlist.count(gti)
            else:
                self.pre_corr_dict[gti] += sum(corri).item()
                self.gt_item_dict[gti] += gtlist.count(gti)

        self.score += (torch.sum(correct) / pre_labels.size(0)).item()
        self.batches += 1

    def score_out(self):
        score_out = self.score / self.batches
        item_score ={}
        for k,v in self.pre_corr_dict.items():
            item_score[k] = v/self.gt_item_dict[k]
        print('test class score:', score_out, item_score)
        return score_out, item_score
