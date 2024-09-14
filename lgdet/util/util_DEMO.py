import torch
import torchvision.utils as vutils
import numpy as np
import torchvision.models as models
from torchvision import datasets
from tensorboardX import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator


def test_tensorboard(write_path):
    writer = SummaryWriter(write_path)
    for n_iter in range(10):
        dummy_s1 = torch.rand(1)
        dummy_s2 = torch.rand(1)
        # data grouping by `slash`
        writer.add_scalar('data/scalar1', dummy_s1[0], n_iter)
        writer.add_scalar('data/scalar2', dummy_s2[0], n_iter)

        writer.add_scalars('data/scalar_group', {'xsinx': n_iter * np.sin(n_iter),
                                                 'xcosx': n_iter * np.cos(n_iter),
                                                 'arctanx': np.arctan(n_iter)}, n_iter)

    writer.export_scalars_to_json(write_path + "/all_scalars.json")
    writer.close()


def read_tbX(path):
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()
    print(ea.scalars.Keys())

    val_acc = ea.scalars.Items('data/scalar1')
    print(len(val_acc))
    print([(i.step, i.value) for i in val_acc])


if __name__ == '__main__':
    p = '../tmp/tbx_log/'
    test_tensorboard(p)
    # tbx_log - -logdir  p
    read_tbX(p)

