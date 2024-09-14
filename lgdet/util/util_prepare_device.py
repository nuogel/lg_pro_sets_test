import torch
import os


def load_device(cfg):
    device = cfg.TRAIN.GPU_NUM
    n_gpu_use = len(device)
    n_gpu = torch.cuda.device_count()
    print('PC gpu numbers:', n_gpu)
    torch.cuda.is_available()
    print('PC gpu is_available:', torch.cuda.is_available())
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        n_gpu_use = 0

    if n_gpu_use > n_gpu:
        print("Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(
            n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device_ids = device[:n_gpu_use]
    device = torch.device('cuda:{}'.format(device[0]) if n_gpu_use > 0 else 'cpu')
    cfg.TRAIN.DEVICE = device
    multi_gpu=True if len(device_ids)>1 else False
    return device, device_ids,multi_gpu
