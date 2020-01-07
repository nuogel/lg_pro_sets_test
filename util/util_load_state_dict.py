import torch


def load_state_dict(model, checkpoint, device):
    new_dic = {}
    state_dict = torch.load(checkpoint, map_location=device)
    for k, v in state_dict.items():
        if 'module.' in k:
            k = k.replace('module.', '')
        new_dic[k] = v
    model.load_state_dict(new_dic)
    return model
