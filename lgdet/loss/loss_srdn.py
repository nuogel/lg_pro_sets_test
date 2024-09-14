import torch.nn as nn
from lgdet.util.util_parse_SR_img import parse_Tensor_img


class SRDNLOSS:
    def __init__(self, cfg):
        self.cfg = cfg
        self.loss_l1loss = nn.L1Loss()  # reduction='sum'
        self.mseloss = nn.MSELoss()  # size_average=False

    def Loss_Call(self, prediction, train_data, losstype='MSE'):
        low_img, high_img, data_infos = train_data

        if self.cfg.TRAIN.SHOW_INPUT:
            parse_Tensor_img(low_img.permute(0, 2, 3, 1), pixcels_norm=self.cfg.TRAIN.PIXCELS_NORM, show_time=self.cfg.TRAIN.SHOW_INPUT)
        if self.cfg.TRAIN.SHOW_TARGET:
            parse_Tensor_img(high_img.permute(0, 2, 3, 1), pixcels_norm=self.cfg.TRAIN.PIXCELS_NORM, show_time=self.cfg.TRAIN.SHOW_TARGET)
        if self.cfg.TRAIN.SHOW_PREDICT:
            parse_Tensor_img(prediction.permute(0, 2, 3, 1), pixcels_norm=self.cfg.TRAIN.PIXCELS_NORM, show_time=self.cfg.TRAIN.SHOW_PREDICT)

        if losstype == 'MSE' or losstype == 'mse':
            loss = self.mseloss(prediction, high_img)
            # mse = loss
        elif losstype == 'L1' or losstype == 'l1':
            loss = self.loss_l1loss(prediction, high_img)
            # mse = self.mseloss(prediction, high_img)
        else:
            loss = None

        # PSNR = 10 * log10(1 / mse.item())
        # print('PSNR: ', PSNR)
        return loss, None
