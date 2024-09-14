import math


class Lr_Sch:
    def __init__(self, cfg):
        ...

    def adjust_learning_rate(self, optimizer, epoch, args):

        lr = self._cosine_decay(args.init_lr, args.final_lr, epoch, args.epochs)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def _cosine_decay(self, init_val, final_val, step, decay_steps):
        alpha = final_val / init_val
        cosine_decay = 0.5 * (1 + math.cos(math.pi * step / decay_steps))
        decayed = (1 - alpha) * cosine_decay + alpha
        return init_val * decayed
