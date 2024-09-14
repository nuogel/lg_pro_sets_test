import torch
from matplotlib import pyplot as plt
import numpy as np

path = 'F:/saved_weight/saved_tbx_log_rdn_youku_200_250_aug_1920_1080x4/checkpoint.pkl'
weight = torch.load(path)

v_all = 0
lt_num_all = 0
cat_array = []
i = 0
lt_thresh = 0.01
show_one_lay = 0

for k, v in weight.items():
    lt_num = 0
    lt_num += torch.sum(torch.abs(v) < lt_thresh)
    lt_num_all += lt_num

    v_array = np.asarray(v.cpu()).reshape(-1)
    idx = np.arange(len(v_array))
    v_all += len(idx)
    if show_one_lay:
        plt.hist(v_array, bins=100)
        plt.xlim(-0.1, 0.1)
        plt.title(str(k) + '---layer num:' + str(i))

        plt.ion()
        plt.pause(1)
        plt.close()
        i += 1
    lowerpersent = float(lt_num) / float(len(v_array))
    print(k, 'value num:', len(v_array), '; less than', lt_thresh, 'is:', lt_num.item(), '; the less percent is:', lowerpersent)
    cat_array.append(lowerpersent)

print(lt_num_all, v_all, 'the less persent is:', float(lt_num_all) / float(v_all))
plt.scatter(np.asarray(range(0, len(cat_array))), np.asarray(cat_array, dtype=np.float32))
plt.title('percent:' + '%.4f' % (float(lt_num_all) / float(v_all)) + ' <' + str(lt_thresh))
plt.show()
# out = np.concatenate(cat_array, 0)
# plt.hist(out, bins=100)
# plt.xlim(-0.3,0.3)
# plt.show()
