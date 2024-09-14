import cv2
import numpy as np
import os
import torch


def _show_img(imgs, labels_out=None, img_in=None, save_labels=False, pic_path=None, show_time=3000, cfg=None, is_training=True, relative_labels=True):
    """
    Show the bounding boxes of images.

    :param imgs: raw images [n, w, h, c]
    :param labels_out: labels out of nms [class, _boxes]
    :return: Show the bounding boxes of images
    """
    font_scale = 0.5
    label_type = "simple"  # "kitti"
    if torch.is_tensor(imgs[0]):
        if imgs.max() <= 1:
            imgs *= 255
        imgs = imgs.numpy().astype(np.uint8)

    if isinstance(imgs, list):
        imgs = np.asarray(imgs)

    if isinstance(labels_out, np.ndarray):
        labels_out = list(labels_out)

    if len(imgs.shape) == 3:
        imgs = imgs[np.newaxis, :]
        labels_out = [labels_out]

    if show_time < 0:
        show_time = 0
    elif is_training and cfg:
        show_time = cfg.TRAIN.SHOW_INPUT
    elif cfg:
        save_labels = cfg.TEST.SAVE_LABELS
        show_time = cfg.TEST.SHOW_TIMES
    if pic_path == None:
        save_labels = False
    imgs_out = []
    for i, _ in enumerate(imgs):
        img_raw = imgs[i]
        labels = labels_out[i]
        if len(labels) > 0:
            txt = ''
            for _, label in enumerate(labels):
                if len(label) == 5:  # in shape of [class, x1, y1, x2, y2]
                    class_out, box = label[0], label[1:5]
                    score_out = 1.0
                elif len(label) == 6:  # in shape of [batch, class, x1, y1, x2, y2]
                    class_out, box = label[1], np.asarray(label[2:6])
                    score_out = label[0]
                elif len(label) == 2:  # [class, bbox[x1,y1,x2,y2]]
                    class_out = label[0]
                    box = label[1]
                    score_out = 1.0
                elif len(label) == 3:  # in shape of [score, class, box[x1, y1, x2, y2]
                    score_out, class_out, box = label[0], label[1], label[2]
                else:
                    print('error: util_show_img-->no such a label shape')
                    continue
                class_out = int(class_out)
                score_out = '%.4f' % score_out
                if cfg:
                    try:
                        class_out = cfg.TRAIN.CLASSES[class_out]
                    except:
                        class_out = str(class_out)
                    rela_lab = relative_labels and cfg.TRAIN.RELATIVE_LABELS
                    if rela_lab:
                        ratio = (img_raw.shape[0], img_raw.shape[1])
                    else:
                        ratio = [1, 1]

                    box[0] *= ratio[1]
                    box[1] *= ratio[0]
                    box[2] *= ratio[1]
                    box[3] *= ratio[0]
                box = [int(box_i) for box_i in box]
                # box[0] = int(box[0])
                # box[1] = int(box[1])
                # box[2] = int(box[2])
                # box[3] = int(box[3])

                img_now = cv2.rectangle(img_raw, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                img_now = cv2.putText(img_now,
                                      str(class_out) + ': ' + score_out,
                                      (int(box[0] + 1), int(box[1] - 7)),
                                      fontFace=cv2.FONT_HERSHEY_COMPLEX,
                                      fontScale=font_scale,
                                      color=(0, 0, 255),
                                      thickness=1)
                if label_type == "kiiti":
                    txt += '{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n' \
                        .format(class_out, 0, 0, 0, box[0], box[1], box[2], box[3], 0, 0, 0, 0, 0, 0, 0, score_out)
                else:
                    txt += '{} {} {} {} {} {}\n'.format(class_out, score_out, box[0], box[1], box[2], box[3])

        else:
            img_now = img_raw
            img_now = cv2.putText(img_now, 'no bounding boxes ...', (10, 10), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                                  fontScale=font_scale, color=(0, 0, 255), thickness=1)

        if save_labels:
            if pic_path is None:
                print('no pic_path, please add it.')
            label_name = os.path.splitext(os.path.split(pic_path)[1])[0]
            label_path = os.path.join(cfg.PATH.GENERATE_LABEL_SAVE_PATH, label_name + '.txt')
            if not os.path.isdir(cfg.PATH.TMP_PATH):
                os.mkdir(cfg.PATH.TMP_PATH)
            if not os.path.isdir(os.path.split(label_path)[0]):
                os.mkdir(os.path.split(label_path)[0])
            # if not os.path.isfile(label_path):
            #     os.mknod(label_path)
            label_file = open(label_path, 'w')
            label_file.write(txt)
            label_file.close()
            print('saved labels to :', label_path)

        if show_time:
            try:
                print('showing pic:', pic_path)
                while img_now.shape[0] > 1920 or img_now.shape[1] > 1920:
                    img_now = cv2.resize(img_now, None, fx=0.8, fy=0.8)
                cv2.imshow('img', img_now)
                cv2.waitKey(show_time)
            except:
                save_path = 'output/' + os.path.basename(pic_path)
                print('can not show img ,so we saved it to: ', save_path)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, img_now)

            # cv2.destroyAllWindows()
        imgs_out.append(img_now)
    return imgs_out


if __name__ == "__main__":
    img = np.array([cv2.imread('E:\LG\programs\lg_pro_sets\datasets/1.jpg')])
    label = [[[1, 20, 30, 50, 60], [2, 150, 160, 180, 190]]]
    _show_img(img, label)
