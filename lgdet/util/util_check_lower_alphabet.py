txt_path = '/media/lg/SSD_WorkSpace/LG/GitHub/lg_pro_sets/datasets/TTS/ALL_XF/ALL_XF_train.txt'


def check_lower(s):
    for c in s:
        if c.islower():
            return True
    return False


def _wrte_dataset_txt(dataset, save_path):
    data_set_txt = ''
    for i in dataset:
        data_set_txt += str(i[0]) + '┣┫' + str(i[1])
    f = open(save_path, 'w', encoding='utf-8')
    f.write(data_set_txt)
    f.close()


f = open(txt_path, 'r', encoding='utf-8')
final = []
i = 0
for line in f.readlines():
    tmp = line.split('┣┫')
    if not check_lower(tmp[-1]):
        final.append(tmp)
    else:
        i += 1
        print(i, line)

final.sort()
save_path = './saved.txt'
_wrte_dataset_txt(final, save_path=save_path)