import torch
import cv2
import math


class Score:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rate_all = 0.
        self.rate_batch = 0.
        self.batches = 0.
        self.mseloss = torch.nn.MSELoss()  # size_average=False

    def init_parameters(self):
        self.rate_all = 0.
        self.rate_batch = 0.
        self.batches = 0.

    def score_out(self):
        score = 0.0
        return score, None, None

