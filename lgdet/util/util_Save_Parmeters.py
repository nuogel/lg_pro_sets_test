"""Saving the parameters while training, and drawing."""
import os
import cv2
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from lgdet.util.util_yml_parse import parse_yaml
from tensorboardX import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator
import shutil
import logging
from argparse import ArgumentParser

LOGGER = logging.getLogger(__name__)


class TxbLogger:
    # TODO: save the weight to parameters.
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.folder = cfg.PATH.TMP_PATH + '/logs/tbx_log_' + cfg.TRAIN.MODEL

    def clean_history_and_init_log(self):
        """Init the parameters."""
        if os.path.isdir(self.folder):
            try:
                shutil.rmtree(self.folder)
            except:
                exit(FileExistsError('FILE IS USING, PLEASE CLOSE IT: {}'.format(self.folder)))
            else:
                info = 'delete the history log from: {}'.format(self.folder)
                LOGGER.info(info)
                print(info)

        self.tbX_writer = SummaryWriter(self.folder)

        self._write_cfg(epoch=0)

    def tbX_write(self, w_dict):
        epoch = w_dict['epoch']

        for k, v in w_dict.items():
            if k == 'epoch' or v is None:
                continue
            if isinstance(v, dict):
                for k1, v1 in v.items():
                    if isinstance(v1, dict):
                        self.tbX_writer.add_scalars(str(k) + '/' + str(k1), v1, epoch)
                    else:
                        self.tbX_writer.add_scalar(str(k) + '/' + str(k1), v1, epoch)
            else:
                self.tbX_writer.add_scalar(str(k), v, epoch)

    def tbX_addImage(self, names, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.tbX_writer.add_image(names, image, 0, dataformats='HWC')

    def tbX_reStart(self, epoch):
        self.tbX_writer = SummaryWriter(self.folder)
        self._write_cfg(epoch=epoch)

    def tbX_show_parameters(self, start_epoch=0, end_epoch=None):
        try:
            ea = event_accumulator.EventAccumulator(self.folder)
            ea.Reload()
            print(ea.scalars.Keys())
        except:
            print('error: no learning_rate in tbX,SET :', self.cfg.TRAIN.LR_START)
            epoch = 0
            learning_rate = self.cfg.TRAIN.LR_START
        else:
            learning_rate = ea.scalars.Items('learning_rate')
            batch_average_loss = ea.scalars.Items('batch_average_loss')
            total_score = ea.scalars.Items('total_score')

            lr = [l_r.value for l_r in learning_rate]
            loss = [l_s.value for l_s in batch_average_loss]
            score = [l_s.value for l_s in total_score]
            self._draw_img([lr, loss, score], start_epoch, end_epoch)

    def _write_cfg(self, epoch=0):
        try:
            f = open(self.cfg.PATH.CLASSES_PATH, 'r', encoding='utf-8')
            lines = f.read().replace('\n', ' ||\t ')
            lines = lines.replace(',', ' <-> ')
            self.tbX_writer.add_text('class_dict', lines, epoch)
        except:
            pass
        config_path = 'cfg/' + self.cfg.BELONGS.lower() + '.yml'
        config_lines = open(config_path, 'r').read().replace('\n', '\n\n')
        self.tbX_writer.add_text('config', config_lines, epoch)

    def save_parameters(self, epoch, learning_rate=None, batch_average_loss=None, f1_score=None, precision=None, recall=None):
        # pylint: disable=too-many-arguments
        """Save the parameters."""
        # check self.file
        if not os.path.isfile(self.file):
            dict_ = {'epoch': 0,
                     'learning_rate': [self.cfg.TRAIN.LR_START],
                     'batch_average_loss': [],
                     'f1_score': [],
                     'precision': [],
                     'recall': []}
            if not os.path.isdir(os.path.dirname(self.file)):
                os.mkdir(os.path.dirname(self.file))
            torch.save(dict_, self.file)
        dict_loaded = torch.load(self.file)
        dict_loaded['epoch'] = epoch
        if learning_rate is not None:
            dict_loaded['learning_rate'] = dict_loaded['learning_rate'][0:epoch + 1]
            dict_loaded['learning_rate'].append(learning_rate)
        if batch_average_loss is not None:
            dict_loaded['batch_average_loss'] = dict_loaded['batch_average_loss'][0:epoch + 1]
            dict_loaded['batch_average_loss'].append(batch_average_loss)
        if f1_score is not None:
            dict_loaded['f1_score'] = dict_loaded['f1_score'][0:epoch + 1]
            dict_loaded['f1_score'].append(f1_score)
        if precision is not None:
            dict_loaded['precision'] = dict_loaded['precision'][0:epoch + 1]
            dict_loaded['precision'].append(precision)
        if recall is not None:
            dict_loaded['recall'] = dict_loaded['recall'][0:epoch + 1]
            dict_loaded['recall'].append(recall)

        torch.save(dict_loaded, self.file)

    def show_parameters(self, start_epoch=0, end_epoch=None):
        """Show the parameters."""

        def _transpose_list(list_in):
            list_out = np.array(list_in).transpose()
            return list_out

        if os.path.isfile(self.file):
            print(self.file)
            dict_loaded = torch.load(self.file)

            learning_rate = dict_loaded['learning_rate']
            batch_average_loss = dict_loaded['batch_average_loss']
            f1_score = dict_loaded['f1_score']
            precision = dict_loaded['precision']
            recall = dict_loaded['recall']
            # print(dict_loaded)

            f1_score = _transpose_list(f1_score)
            precision = _transpose_list(precision)
            recall = _transpose_list(recall)

            self._draw_img([learning_rate, batch_average_loss, f1_score, precision, recall], start_epoch, end_epoch)
        else:
            print('no parameter.file is found.')

    def _draw_img(self, datas, start_epoch=0, end_epoch=None):
        """Show the parameters."""

        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'brown', 'darkred', 'linen']
        linestyles = ['-', '-.', '--', ':', '-', '-.', '--', ':', '-', '-.', '--', ':', ]

        labels = ['+', 'o', '*', '.', 'x', 'p', 'H', 'h', '^', '>', '<', '1', '2', '3', '4']
        line_illustration = ['learning_rate', 'batch_average_loss', 'score', 'precision', 'recall']

        for i, data in enumerate(datas):
            if data is []:  # if there is no data then continue
                continue
            end_epoch = len(data) - 1 if not end_epoch else end_epoch
            data_x1 = np.array(range(len(data)))[start_epoch:end_epoch + 1]
            data_y1 = np.array(data, dtype=np.float32)[start_epoch:end_epoch + 1]
            plt.figure(num=1, figsize=(20, 10))
            plt.subplot(321 + i)
            plt.xlabel('Epoch')
            plt.ylabel(line_illustration[i])
            for ii, da in enumerate(data_y1):
                print(ii, '->', da)
            if len(data_y1) == 0:  # if there is no data in it ,then do nothing.
                continue
            if data_y1.shape.__len__() == 1:
                plt.plot(data_x1, data_y1, color=colors[i], linewidth=3,
                         linestyle=linestyles[i], label=labels[i])
                # print(data_x1[-1], data_y1[-1])
                plt.text(x=data_x1[-1], y=data_y1[-1], s='(' + str(data_x1[-1]) + ',' + str(data_y1[-1]) + ')')
            else:
                data_x = range(data_y1.shape[1])[start_epoch:end_epoch + 1]
                for j, data_y in enumerate(data_y1):
                    plt.plot(data_x, data_y[data_x], color=(colors[j]),
                             linewidth=3,
                             linestyle=linestyles[j],
                             label=self.cfg.TRAIN.CLASSES[j] + str('(%0.3f)' % data_y[data_x[-1]]))
                    plt.legend(loc='lower right')

        plt.show()


if __name__ == "__main__":
    """Test show_parameters."""


    def _parse_arguments():
        parser = ArgumentParser()
        parser.add_argument('--type', default='obd', type=str, help='yml_path')
        return parser.parse_args()


    args = _parse_arguments()
    cfg = parse_yaml(args)
    cfg.PATH.TMP_PATH = os.path.join('../..', cfg.PATH.TMP_PATH)
    para = TxbLogger(cfg)
    para.tbX_show_parameters(start_epoch=0, end_epoch=None)
