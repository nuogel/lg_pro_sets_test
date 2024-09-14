import numpy as np
import cv2
import os
import time
from argparse import ArgumentParser
from others.track.KCF.example1.kcf_tracker import Kcftracker


def display_tracker_lg(img, bbox):
    img = cv2.imread(img)
    visual(img, bbox)


def visual(img, bbox):
    (x, y, w, h) = bbox
    x1 = int(x)
    y1 = int(y)
    w = int(w)
    h = int(h)
    pt1, pt2 = (x1, y1), (x1 + w, y1 + h)
    img_rec = cv2.rectangle(img, pt1, pt2, (0, 255, 255), 2)
    cv2.imshow('window', img_rec)
    cv2.waitKey()


def load_bbox(ground_file, resize, dataformat=0):
    f = open(ground_file)
    lines = f.readlines()
    bbox = []
    for line in lines:
        if line:
            pt = line.strip().split(',')
            pt_int = [float(ii) for ii in pt]
            bbox.append(pt_int)
    bbox = np.array(bbox)
    if resize:
        bbox = (bbox.astype('float32') / 2).astype('int')
    else:
        bbox = bbox.astype('float32').astype('int')
    if dataformat:
        bbox[:, 2] = bbox[:, 0] + bbox[:, 2]
        bbox[:, 3] = bbox[:, 1] + bbox[:, 3]
    return bbox


def load_imglst(img_dir):
    file_lst = [pic for pic in os.listdir(img_dir) if '.jpg' in pic]
    img_lst = [os.path.join(img_dir, filename) for filename in file_lst]
    return img_lst


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--dataset_descriptor', type=str, default='E:/datasets/TRACK/OTB100/BlurCar1/',
                        help='The directory of video and groundturth file')
    parser.add_argument('--save_directory', type=str, default='./',
                        help='The directory of result file')
    parser.add_argument('--show_result', type=int,
                        help='Show result or not', default=1)

    return parser.parse_args()


def main(args):
    # Load  arg
    dataset = args.dataset_descriptor
    save_directory = args.save_directory
    show_result = args.show_result
    padding = 2
    dataformat = 1

    # Load dataset information and get start position
    title = dataset.split('/')
    title = [t for t in title if t][-1]
    img_lst = load_imglst(dataset + '/img/')
    bbox_lst = load_bbox(os.path.join(dataset + '/groundtruth_rect.txt'), 0, dataformat=0)
    px1, py1, px2, py2 = bbox_lst[0]
    pos = (px1, py1, px2, py2)
    frames = len(img_lst)
    # Attention: the original data format is (y,x,h,w), so the  code above translate
    # the data to (x1,y1,x2,y2) format

    # Create file to record the result
    tracker_bb = []
    result_file = os.path.join(save_directory, title + '_' + 'result_KCF.txt')
    file = open(result_file, 'w')
    start_time = time.time()

    # Tracking
    for i in range(frames):
        img = cv2.imread(img_lst[i])
        if i == 0:
            # Initialize trakcer, img 3 channel, pos(x1,y1,x2,y2)
            kcftracker = Kcftracker(img, pos, HOG_flag=1, dataformat=0, resize=0)
        else:
            # Update position and traking
            pos = kcftracker.updateTracker(img)

        # Write the position
        out_pos = pos  # [pos[1], pos[0], pos[3] - pos[1], pos[2] - pos[0]]
        win_string = [str(p) for p in out_pos]
        win_string = ",".join(win_string)
        tracker_bb.append(win_string)
        file.write(win_string + '\n')

        if show_result:
            display_tracker_lg(img_lst[i], out_pos)

    duration = time.time() - start_time
    fps = int(frames / duration)
    print('each frame costs %3f second, fps is %d' % (duration / frames, fps))
    file.close()
    result = load_bbox(result_file, 0)

    # # Show the result with bbox
    # if show_result:
    #     display_tracker(img_lst, result, save_flag=0)


cfg = parse_arguments()
main(cfg)
