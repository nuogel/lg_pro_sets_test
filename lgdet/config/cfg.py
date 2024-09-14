import os
import torch
from lgdet.util.util_Save_Parmeters import TxbLogger
from lgdet.util.util_logger import load_logger
import numpy as np


def prepare_cfg(cfg, args, is_training=True):
    torch.backends.cudnn.benchmark = True
    os.makedirs(cfg.PATH.TMP_PATH, exist_ok=True)

    print('torch version: ', torch.__version__)
    print('torch.version.cuda: ', torch.version.cuda)

    if cfg.TRAIN.MODEL in ['SRDN', 'srdn', 'tacotron2']:  # , 'efficientdet']:
        cfg.manual_seed = True
    else:
        cfg.manual_seed = False

    try:
        if args.model:
            cfg.TRAIN.MODEL = args.model
        if args.data_path:
            cfg.PATH.INPUT_PATH = args.data_path
        if args.gpu:
            cfg.TRAIN.GPU_NUM = args.gpu
        if args.score_thresh:
            cfg.TEST.SCORE_THRESH = args.score_thresh
        cfg.TRAIN.pre_trained = args.pre_trained
    except:
        pass

    if is_training:
        if args.epoch_size:
            cfg.TRAIN.EPOCH_SIZE = args.epoch_size
        if args.test_only:
            cfg.TEST.TEST_ONLY = args.test_only

        cfg.TRAIN.EMA = args.ema
        cfg.TRAIN.AMP = args.autoamp
        cfg.TEST.MAP_FSCORE = args.map_fscore

    cfg.checkpoint = args.checkpoint
    cfg = common_cfg(cfg)

    cfg.writer = TxbLogger(cfg)
    cfg.logger = load_logger(cfg, args)

    if args.batch_size != 0:
        cfg.TRAIN.BATCH_SIZE = args.batch_size
        assert cfg.TRAIN.BATCH_SIZE > 0, 'batch size <0 !!!!'
    if cfg.TEST.ONE_TEST:
        cfg.TRAIN.RESIZE = 1
        cfg.TRAIN.LETTERBOX = 0
        cfg.TRAIN.DO_AUG = 0
        cfg.TRAIN.GRAY_BINARY = 0
        cfg.TRAIN.USE_LMDB = 0
        cfg.TRAIN.MOSAIC = 0
        cfg.TRAIN.MULTI_SCALE = 0
        cfg.TRAIN.WARM_UP_STEP = 0

        cfg.TEST.LETTERBOX = 0
        cfg.TEST.RESIZE = 1
        cfg.TEST.MULTI_SCALE = 0

    try:
        if cfg.TEST.ONE_TEST:
            args.number_works = 0
    except:
        pass

    if cfg.BELONGS in ['imc', ]:
        try:
            cfg.PATH.CLASSES_PATH = cfg.PATH.CLASSES_PATH.format(cfg.TRAIN.TRAIN_DATA_FROM_FILE[0].lower())
        except:
            print('class names error:', cfg.PATH.CLASSES_PATH)
        try:
            from lgdet.util.util_get_cls_names import _get_class_names
            class_dict = _get_class_names(cfg.PATH.CLASSES_PATH)
            class_names = []
            for k, v in class_dict.items():
                if v not in class_names:
                    class_names.append(v)
            cfg.TRAIN.CLASSES_NUM = len(class_names)
            cfg.TRAIN.CLASSES = class_names
        except EnvironmentError:
            print('cfg.py trying get class number and classes faild.')


    elif cfg.BELONGS in ['obd', ]:
        if cfg.TEST.MAP_FSCORE == 0:
            cfg.TEST.SCORE_THRESH = 0.05
        # single level anchor box config for VOC and COCO
        ANCHOR_SIZE = [[1.19, 1.98], [2.79, 4.59], [4.53, 8.92], [8.06, 5.29], [10.32, 10.65]]

        ANCHOR_SIZE_COCO = [[0.53, 0.79], [1.71, 2.36], [2.89, 6.44], [6.33, 3.79], [9.03, 9.74]]

        # multi level anchor box config for VOC and COCO
        # yolo_v3
        MULTI_ANCHOR_SIZE = [[32.64, 47.68], [50.24, 108.16], [126.72, 96.32],
                             [78.4, 201.92], [178.24, 178.56], [129.6, 294.72],
                             [331.84, 194.56], [227.84, 325.76], [365.44, 358.72]]

        MULTI_ANCHOR_SIZE_COCO = [[12.48, 19.2], [31.36, 46.4], [46.4, 113.92],
                                  [97.28, 55.04], [133.12, 127.36], [79.04, 224.],
                                  [301.12, 150.4], [172.16, 285.76], [348.16, 341.12]]

        # tiny yolo_v3
        TINY_MULTI_ANCHOR_SIZE = [[34.01, 61.79], [86.94, 109.68], [93.49, 227.46],
                                  [246.38, 163.33], [178.68, 306.55], [344.89, 337.14]]

        TINY_MULTI_ANCHOR_SIZE_COCO = [[15.09, 23.25], [46.36, 61.47], [68.41, 161.84],
                                       [168.88, 93.59], [154.96, 257.45], [334.74, 302.47]]

        # anchor_yolov2 = [[373, 326],  # [W,H]
        #                  [156, 198],
        #                  [116, 90],
        #                  [59, 119],
        #                  [62, 45],
        #                  [30, 61],
        #                  [33, 23],
        #                  [16, 30],
        #                  [10, 13]]

        ICONS_ANCHORS = [[61, 183], [15, 36], [29, 19], [25, 13], [18, 13], [10, 17]]

        anchor_yolov2 = [[185., 112.],
                         [87., 32.],
                         [25., 83.],
                         [60., 26.],
                         [26., 26.],
                         [17., 36.],
                         [37., 14.],
                         [27., 12.],
                         [12., 14.]]
        # anchor_yolov2 = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]

        anchor_yolov3_tiny = [[344.89, 337.14], [178.68, 306.55], [246.38, 163.33],
                              [93.49, 227.46], [86.93, 109.69], [34.01, 61.78]]  # others

        anchor_yolov3 = [[331.84, 194.56], [227.84, 325.76], [365.44, 358.72],
                         [78.4, 201.92], [178.24, 178.56], [129.6, 294.72],
                         [32.64, 47.68], [50.24, 108.16], [126.72, 96.32], ]

        anchor_yolov3_voc_auto = [[516, 534], [454, 302], [250, 456], [240, 236], [127, 309], [118, 134], [63, 183], [52, 80], [27, 50]]
        # yolov3 writer's anchors: [10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326]
        anchor_yolov3_writer = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]

        VISDRONE_anchors = [[104., 92.],
                            [57., 41.],
                            [27., 43.],
                            [31., 20.],
                            [15., 23.],
                            [8., 11.],
                            ]
        # anchors should be 倒序。
        try:
            cfg.PATH.CLASSES_PATH = cfg.PATH.CLASSES_PATH.format(cfg.TRAIN.TRAIN_DATA_FROM_FILE[0].lower())
        except:
            print('class names error:', cfg.PATH.CLASSES_PATH)

        cfg.TRAIN.ANCHORS = anchor_yolov3
        if cfg.TEST.ONE_TEST:
            # if cfg.TEST.ONE_NAME != []:
            #     cfg.TRAIN.BATCH_SIZE = len(cfg.TEST.ONE_NAME)
            # else:
            cfg.TRAIN.BATCH_SIZE = args.batch_size
            cfg.TRAIN.BATCH_BACKWARD_SIZE = 1

        if 'yolov3_tiny' in cfg.TRAIN.MODEL and ('KITTI' in cfg.TRAIN.TRAIN_DATA_FROM_FILE):
            cfg.TRAIN.FMAP_ANCHOR_NUM = 3
            cfg.TRAIN.ANCHORS = anchor_yolov3_tiny
        elif 'yolov3_tiny' in cfg.TRAIN.MODEL and (cfg.TRAIN.TRAIN_DATA_FROM_FILE[0] in ['VISDRONE', 'AUTOAIR']):
            cfg.TRAIN.FMAP_ANCHOR_NUM = 3
            cfg.TRAIN.ANCHORS = VISDRONE_anchors
        if 'yolov3_tiny' in cfg.TRAIN.MODEL:
            cfg.TRAIN.FMAP_ANCHOR_NUM = 3
            cfg.TRAIN.ANCHORS = anchor_yolov3_tiny
        elif cfg.TRAIN.MODEL in ['yolov3', 'yolonano']:
            cfg.TRAIN.FMAP_ANCHOR_NUM = 3
            cfg.TRAIN.ANCHORS = anchor_yolov3
        elif cfg.TRAIN.MODEL in ['yolov5']:
            cfg.TRAIN.FMAP_ANCHOR_NUM = 3
            cfg.TRAIN.ANCHORS = anchor_yolov3_writer
        elif 'yolov2' in cfg.TRAIN.MODEL:
            cfg.TRAIN.FMAP_ANCHOR_NUM = len(anchor_yolov2)
            cfg.TRAIN.ANCHORS = anchor_yolov2

        check_anchors(cfg)

        try:
            from lgdet.util.util_get_cls_names import _get_class_names
            classpath = os.path.join(os.path.dirname(__file__),'../..',cfg.PATH.CLASSES_PATH)
            class_dict = _get_class_names(classpath)
            class_names = []
            for k, v in class_dict.items():
                if v not in class_names:
                    class_names.append(v)
            cfg.TRAIN.CLASSES_NUM = len(class_names)
            cfg.TRAIN.CLASSES = class_names
        except EnvironmentError:
            AttributeError('cfg.py trying get class number and classes faild.')

        ## config for FCOS:
        # backbone
        if cfg.TRAIN.MODEL == 'fcos':
            cfg.TRAIN.RELATIVE_LABELS = 0
            cfg.pretrained = True
            cfg.freeze_stage_1 = False
            cfg.freeze_bn = False

            # fpn
            cfg.fpn_out_channels = 256
            cfg.use_p5 = True

            # head
            cfg.use_GN_head = True
            cfg.prior = 0.01
            cfg.add_centerness = True
            cfg.cnt_on_reg = True

            # training
            cfg.strides = [8, 16, 32, 64, 128]
            cfg.limit_range = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 999999]]

            # inference
            cfg.score_threshold = 0.5
            cfg.nms_iou_threshold = 0.5
            cfg.max_detection_boxes_num = 1000

    return cfg, args


def common_cfg(cfg):
    mean = [0.485, 0.456, 0.406]  # RGB
    std = [0.229, 0.224, 0.225]
    mean = np.asarray(mean, np.float32)
    std = np.asarray(std, np.float32)
    cfg.mean = mean
    cfg.std = std
    return cfg


def check_anchors(cfg):
    areas = []
    for anc in cfg.TRAIN.ANCHORS:
        areas.append(anc[0] * anc[1])

    if areas[0]<areas[-1]:
        cfg.TRAIN.ANCHORS = cfg.TRAIN.ANCHORS[::-1]