import torch
import torch.nn


def to_onehot(lab, length):
    emb = torch.nn.Embedding(length, length)
    emb.weight.data = torch.eye(length)
    out = emb(lab.type(torch.LongTensor))
    return out


if __name__ == '__main__':
    lab = [0, 1, 4, 7, 3, 2]
    lab = torch.LongTensor(lab)
    print(to_onehot(lab, 10))
