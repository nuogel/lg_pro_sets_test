"""Calculate the F1 score."""
from lgdet.registry import SCORES


@SCORES.registry()
class Score:
    """Calculate the F1 score."""

    def __init__(self, cfg):
        """Init the parameters."""
        if cfg.TEST.MAP_FSCORE:
            from .obd_score_base.f1score import FnScore
            self.score = FnScore(cfg)
        else:
            from .obd_score_base.map import MAP
            self.score = MAP(cfg)

    def init_parameters(self):
        self.score.init_parameters()

    def cal_score(self, pre_labels, test_data=None, from_net=True):
        self.score.cal_score(pre_labels, test_data, from_net)

    def score_out(self):
        return self.score.score_out()
