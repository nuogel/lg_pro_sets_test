#! /usr/bin/env python3
"""Script to test the apollo yolo."""

import sys
from argparse import ArgumentParser
from util.util_yml_parse import parse_yaml
from net_works.test_solver import Test


def _parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--yml_path', default='cfg/OBD.yml'  # OBD SR_DN
                        , type=str, help='yml_path')
    parser.add_argument('--checkpoint', default=
                        # 'tmp/checkpoint/now.pkl'
                        # 'F:/test_results/saved_tbx_log_yolov3_tiny_clean/checkpoint.pkl',
                        'F:/test_results/tbx_log_ssd_coco/checkpoint.pkl'
                        , help='Path to the checkpoint to be loaded to the model')
    return parser.parse_args()


def main():
    """Run the script."""
    exit_code = 0
    # file_s = 'dataset//0853_L.png'
    # file_s = 'dataset//00000_L.png'
    # file_s = 'E:/datasets/Noise_Images/NOISE_kitti/level_2/images/000002.png'
    # file_s = 'E://datasets//kitti//training//image_2//'
    # # file_s = 'datasets/kitti/training/image_2/000000.png'
    # file_s = 'E:/datasets/Noise_Images/NOISE_kitti/level_5/images/KITTI_005066_1.png'
    # file_s = 'evasys/voc_mAP/occ_good.txt'
    # file_s = 'E:/datasets/NLP/THCHS/data_thchs30/data/A2_0.wav'  # /'
    # file_s = 'E:/datasets/VOCdevkit/VOC2007/JPEGImages/000207.jpg'
    # file_s = 'E:/LG/GitHub/lg_pro_sets/tmp/idx_stores/occ_2.txt'
    file_s = 'tmp/idx_stores/test_set_coco_car.txt'
    # file_s = 'E:/datasets/Car/COCO_Car/images/COCO_train2014_000000036639.jpg'
    args = _parse_arguments()
    cfg = parse_yaml(args.yml_path)
    # file_s = cfg.TEST.ONE_NAME[0]
    test = Test[cfg.BELONGS](cfg, args)
    test.test_run(file_s)
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
