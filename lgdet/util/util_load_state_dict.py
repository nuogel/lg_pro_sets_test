import torch
from collections import OrderedDict


def load_state_dict(model, checkpoint, device):
    new_dic = OrderedDict()
    checkpoint = torch.load(checkpoint, map_location=device)
    state_dict = checkpoint['state_dict']
    last_epoch = checkpoint['epoch']
    optimizer = checkpoint['optimizer']
    global_step = checkpoint['global_step']
    for k, v in state_dict.items():
        if 'module.' == k[:8]:
            k = k.replace('module.', '')
        new_dic[k] = v
    model.load_state_dict(new_dic)
    return model, last_epoch, optimizer, global_step
