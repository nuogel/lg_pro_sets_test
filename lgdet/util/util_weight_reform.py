import torch
import os
from collections import OrderedDict


def weight16_17():
    weight_path_16 = '../../tmp/checkpoints/lrf300/now.pkl'
    name_17 = os.path.basename(weight_path_16).split('.')[0]
    weight_path_17 = os.path.join(os.path.dirname(weight_path_16), name_17 + '_17.pkl')
    weight_16 = torch.load(weight_path_16)
    torch.save(weight_16, weight_path_17, _use_new_zipfile_serialization=False)


def weight_discard_module():
    checkpoint = '/media/lg/SSD_WorkSpace/LG/OBD/FCOS-PyTorch-37.2AP/checkpoint/voc_77.8.pth'
    name = os.path.basename(checkpoint).split('.')[0]
    new_path = os.path.join(os.path.dirname(checkpoint), name + '_without_module.pkl')
    new_dic = OrderedDict()
    state_dict = torch.load(checkpoint)
    for k, v in state_dict.items():
        if 'module.' == k[:7]:
            k = k.replace('module.', '')
        new_dic[k] = v
    torch.save(new_dic, new_path)
    print('saved to ', new_path)


def weight_reshape():
    checkpoint = '/media/lg/SSD_WorkSpace/LG/GitHub/lg_pro_sets/saved/checkpoint/fcos_voc_77.8.pkl'
    name = os.path.basename(checkpoint).split('.')[0]
    new_path = os.path.join(os.path.dirname(checkpoint), name + '_new.pkl')
    new_dic = OrderedDict()
    state_dict = torch.load(checkpoint)
    old_name = 'fcos_body.'
    new_name = ''
    for k, v in state_dict.items():
        if old_name in k:
            k = k.replace(old_name, new_name)
        new_dic[k] = v
    torch.save(new_dic, new_path)
    print('saved to ', new_path)


def decode_weight_lg():
    checkpoint = '/media/lg/SSD_WorkSpace/LG/GitHub/lg_pro_sets/tmp/checkpoints/tacotron2/now.pkl'
    state_dict = torch.load(checkpoint)

    name = os.path.basename(checkpoint).split('.')[0]
    new_path = os.path.join(os.path.dirname(checkpoint), name + '_new.pkl')

    model = state_dict['state_dict']

    new_dic = {
        'model': model,
        'epoch': 10
    }
    torch.save(new_dic, new_path)
    print('saved to ', new_path)

def analies_checkpoint():
    path = '/home/dell/lg/code/lg_pro_sets/tmp/checkpoints/yolov5/now.pkl'
    checkpoint = torch.load(path)
    state_dict = checkpoint['state_dict']
    layer_i = 0
    for k,v in state_dict.items():
        layer_i +=1
        print(layer_i, ':', k, '------',v.shape)


if __name__ == '__main__':
    # weight_discard_module()
    # weight_reshape()
    # decode_weight_lg()
    analies_checkpoint()