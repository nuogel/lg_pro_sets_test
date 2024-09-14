# coding=utf-8
#!/usr/bin/env Python3
"""Script to train the apollo yolo model."""

import sys

# print(sys.path)
# sys.path.append('/home/lg/new_disk/deep_learning/lg_Pro_Set')
from argparse import ArgumentParser
from lgdet.solver.train_solver import Solver
from lgdet.util.util_yml_parse import parse_yaml


def _parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--type', default='obd', type=str, help='yml_path')
    parser.add_argument('--model', type=str, help='yml_path')
    parser.add_argument('--checkpoint', '--cp', default=1
                        , help='Path to the checkpoint to be loaded to the model')
    parser.add_argument('--data_path', '--dp', help='batch size')
    parser.add_argument('--epoch_size', '--ep', type=int, help='batch size')
    parser.add_argument('--batch_size', '--bz', default=4, type=int, help='batch size')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--lr_continue', '--lr_c', default=0, type=float, help='learning rate')
    parser.add_argument('--number_works', '--nw', default=0, type=int, help='number works of dataloader')
    parser.add_argument('--score_thresh', '--st', default=0.25, type=float, help='score_thresh')
    parser.add_argument('--ema', default=1, type=int, help='ema')
    parser.add_argument('--autoamp', default=0, type=int, help='autoamp')

    parser.add_argument('--gpu', help='number works of dataloader')
    parser.add_argument('--pre_trained', '--pt', default=2
                        ,help='1:loading pre_trained from path, 2:loading pre_trained from model.weight_inits() itself')
    parser.add_argument('--map_fscore', '--mf', default=0, type=int, help='map-0; fcore-1')

    parser.add_argument('--epoch_continue', default=None, type=int, help='Epoch of continue training')
    parser.add_argument('--debug', '--d', action='store_true', default=False, help='Enable verbose info')
    parser.add_argument('--test_only', '--to', default=0, type=int, help='test only')
    return parser.parse_args()


def main():
    """Main. entry of this script."""
    exit_code = 0
    args = _parse_arguments()
    print(args)
    cfg = parse_yaml(args)
    solver = Solver(cfg, args, train=True)
    solver.train()
    # try:
    #     solver.train()
    # except:
    #     print('warning: please do not close this, saving checkpoints......')
    #     solver._save_checkpoint()
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
