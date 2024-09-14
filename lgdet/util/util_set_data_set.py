"""This code is used to set data set manually"""
import torch
from sklearn.model_selection import train_test_split


def add_new_idx():
    path = '../../tmp/idx_stores/'
    train_set_old = torch.load(path+'train_set')
    test_set_old = torch.load(path+'test_set')
    idx_train = range(2374, 4068)
    train_set_add, test_set_add = train_test_split(idx_train, test_size=0.1)
    train_set_new = train_set_old + train_set_add
    test_set_new = test_set_old + test_set_add
    torch.save(train_set_new, path+'train_set')
    torch.save(test_set_new, path+'test_set')


if __name__ == "__main__":
    print('be care of running this code...')
    # add_new_idx()

