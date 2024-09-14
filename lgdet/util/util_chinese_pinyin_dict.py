import torch
from pypinyin import lazy_pinyin, Style

style = Style.TONE3


def make_dict_BZNSYP():
    path = '/media/lg/DataSet_E/datasets/BZNSYP/bbspeech/train.txt'
    py = []
    for line in open(path, 'r', encoding='utf-8').readlines():
        tmp = line.split('|')[-1]
        txt = tmp.strip().replace(' ', '')
        txt_yj = lazy_pinyin(txt, style)
        py += txt_yj
    py = list(set(py))
    return py


def make_dict_AISHELL1():
    path = '/media/lg/DataSet_E/datasets/Aishell_1/transcript/aishell_transcript_v0.8.txt'
    py = []
    for line in open(path, 'r', encoding='utf-8').readlines():
        tmp = line.split(' ', 1)
        txt = tmp[1].strip().replace(' ', '')
        txt_yj = lazy_pinyin(txt, style)
        py += txt_yj
    py = list(set(py))
    return py


def make_dict(datasets=['BZNSYP', 'AISHELL1']):
    py_all = []
    for dataset in datasets:
        if dataset == 'BZNSYP':
            py_bznsyp = make_dict_BZNSYP()
            py_all += py_bznsyp

        elif dataset == 'AISHELL1':
            by_aishell1 = make_dict_AISHELL1()
            py_all += by_aishell1

        else:
            print('no such a dataset')

    simbols = ['，', '。', '；', '？', '《', '》', '、', '！', '@', '%', '(', ')', '：', '“ ', '”']
    py_all += simbols
    py_all = list(set(py_all))
    py_all.sort()
    py_all.insert(0, '~')
    py_all.insert(0, '_')

    dict_py = dict(zip(py_all, range(len(py_all))))
    torch.save(dict_py, 'pinyin_dict.pth')


if __name__ == '__main__':
    make_dict(datasets=['BZNSYP', 'AISHELL1'])
    ict_py = torch.load('pinyin_dict.pth')
    a = 1
