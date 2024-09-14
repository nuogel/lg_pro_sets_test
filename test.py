#! /usr/bin/env python3
"""Script to test the apollo yolo."""

import sys
from argparse import ArgumentParser
from lgdet.util.util_yml_parse import parse_yaml
from lgdet.solver.test_pakage.test_asr import Test_ASR
from lgdet.solver.test_pakage.test_obd import Test_OBD
from lgdet.solver.test_pakage.test_srdn import Test_SRDN
from lgdet.solver.test_pakage.test_tts import Test_TTS
from lgdet.solver.test_pakage.test_vid import Test_VID


def _parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--type', default='obd'  # SR_DN
                        , type=str, help='yml_path')
    parser.add_argument('--model', type=str, help='yml_path')
    parser.add_argument('--checkpoint', '--cp', default=1
                        , help='Path to the checkpoint to be loaded to the model')
    parser.add_argument('--pre_trained', '--pt', default=0  # 'saved/checkpoint/fcos_voc_77_new.pkl'
                        , help='Epoch of continue training')
    parser.add_argument('--batch_size', '--bz', default=1, type=int, help='batch size')
    parser.add_argument('--gpu', help='number works of dataloader')
    parser.add_argument('--ema', default=0, type=int, help='ema')
    parser.add_argument('--number_works', '--nw', default=0, type=int, help='number works of dataloader')
    parser.add_argument('--debug', '--d', action='store_true', default=False, help='Enable verbose info')
    parser.add_argument('--score_thresh', '--st', default=0.6, type=float, help='score_thresh')

    return parser.parse_args()


def main():
    """Run the script."""
    exit_code = 0
    files = 'one_name'
    # files = 'test_set'
    # files = 'datasets/OBD/VOC/VOC_test.txt'
    # files = 'datasets/OBD_idx_stores/COCO/COCO_test.txt'  #
    # files = 'datasets/OBD_idx_stores/KITTI/KITTI_test.txt'
    # files = 'datasets/TTS/BZNSYP/BZNSYP_test.txt'
    score = False
    args = _parse_arguments()
    cfg = parse_yaml(args)
    Test = {'obd': Test_OBD, 'asr': Test_ASR, 'srdn': Test_SRDN, 'vid': Test_VID, 'tts': Test_TTS, }
    test = Test[cfg.BELONGS](cfg, args, train=None)
    test.test_run(files)
    if score:
        test.score(txt_info=files, pre_path=cfg.PATH.GENERATE_LABEL_SAVE_PATH)
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
