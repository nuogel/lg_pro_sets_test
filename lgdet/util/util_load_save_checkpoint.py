import os
import torch
from collections import OrderedDict


def _load_checkpoint(model, checkpoint, device):
    new_dic = OrderedDict()
    checkpoint = torch.load(checkpoint, map_location=device)
    state_dict = checkpoint['state_dict']
    last_epoch = checkpoint['epoch']
    optimizer = checkpoint['optimizer']
    optimizer_type = checkpoint['optimizer_type']
    global_step = checkpoint['global_step']

    for k, v in state_dict.items():
        if 'module.' == k[:7]:
            k = k.replace('module.', '')
        new_dic[k] = v

    model.load_state_dict(new_dic)
    return model, last_epoch, optimizer, optimizer_type, global_step


def _load_pretrained(model, pre_trained, device):
    new_dic = OrderedDict()
    checkpoint = torch.load(pre_trained, map_location=device)
    try:
        state_dict = checkpoint['state_dict']
    except:
        try:
            state_dict = checkpoint['model']
        except:
            state_dict = checkpoint

    for k, v in state_dict.items():
        if 'module.' == k[:7]:
            k = k.replace('module.', '')
        new_dic[k] = v

    try:
        ret = model.load_state_dict(new_dic, strict=False)
        print(ret)
    except RuntimeError as e:
        print('Ignoring ' + str(e) + '"')

    return model


def _save_checkpoint(self):
    _model = self.ema.ema if self.cfg.TRAIN.EMA else self.model
    saved_dict = {'epoch': self.epoch,
                  'state_dict': _model.state_dict(),
                  'optimizer': self.optimizer.state_dict(),
                  'optimizer_type': self.cfg.TRAIN.OPTIMIZER.lower(),
                  'global_step': self.global_step}

    path_list = [str(self.epoch), 'now']
    for path_i in path_list:
        checkpoint_path = os.path.join(self.cfg.PATH.TMP_PATH, 'checkpoints/' + self.cfg.TRAIN.MODEL, path_i + '.pkl')
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(saved_dict, checkpoint_path)
        print('checkpoint is saved:', checkpoint_path)
