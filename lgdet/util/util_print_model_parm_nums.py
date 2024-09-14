from torchsummary import summary


def _print_model_parm_nums(Model, input_size_W, input_size_H, modelName=None, ):
    print('Using Model Net:', modelName)
    summary(model=Model.cuda(), input_size=((input_size_W, input_size_H, 3), 0))
    total = sum([param.nelement() for param in Model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))


if __name__ == '__main__':
    _print_model_parm_nums(self.model.to(self.cfg.TRAIN.DEVICE), self.cfg.TRAIN.IMG_SIZE[0], self.cfg.TRAIN.IMG_SIZE[1])
