import os
import torch
from torch.optim import lr_scheduler
from util.util_ConfigFactory_Classes import get_loss_class, get_score_class
from util.util_prepare_device import load_device
from cfg.cfg import prepare_cfg
from lgdet.dataloader.DataLoaderFactory import DataLoaderFactory
from lgdet.registry import MODELS, LOSSES, SCORES, build_from_cfg
from util.util_weights_init import weights_init
from util.util_get_dataset_from_file import _read_train_test_dataset
from collections import OrderedDict
from lgdet.metrics.ema import ModelEMA
import math
from matplotlib import pyplot as plt


class BaseSolver(object):
    def __init__(self, cfg, args, train):
        self.is_training = train
        self._get_configs(cfg, args)
        self._get_model()
        self._get_dataloader()
        self.epoch = 0

    def _get_configs(self, cfg, args):
        self.cfg, self.args = prepare_cfg(cfg, args)
        self.cfg.TRAIN.DEVICE, self.device_ids = load_device(self.cfg)

    def _get_model(self):
        self.ema = 0
        self.model = build_from_cfg(MODELS, str(self.cfg.TRAIN.MODEL).upper())(self.cfg)
        # init model:
        if self.args.checkpoint not in [0, '0', 'None', 'no', 'none', "''"]:
            if self.args.checkpoint in [1, '1']: self.args.checkpoint = os.path.join(self.cfg.PATH.TMP_PATH + 'checkpoints/' + self.cfg.TRAIN.MODEL, 'now.pkl')
            self.model, self.epoch_last, self.optimizer_dict, self.global_step = self._load_checkpoint(self.model, self.args.checkpoint, self.cfg.TRAIN.DEVICE,
                                                                                                       self.args.pre_trained)
            self.cfg.writer.tbX_reStart(self.epoch_last)
        else:
            self.model = weights_init(self.model, self.cfg)
            self.optimizer_dict = None
            self.epoch_last = 0
            self.global_step = 0
            self.cfg.writer.clean_history_and_init_log()

        if len(self.device_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)

        if self.cfg.TRAIN.EMA:
            self.ema = ModelEMA(self.model, device=self.cfg.TRAIN.DEVICE)

        if self.is_training:
            self.model.train()
        else:
            if not self.cfg.TEST.ONE_TEST:
                self.model.eval()
            else:
                self.model.train()

        self.model = self.model.to(self.cfg.TRAIN.DEVICE)

    def _get_score(self):
        self.score = get_score_class(self.cfg.BELONGS)(self.cfg)

    def _get_lossfun(self):
        self.lossfun = get_loss_class(self.cfg.BELONGS, self.cfg.TRAIN.MODEL)(self.cfg)

    def _get_optimizer(self):
        opt_type = self.cfg.TRAIN.OPTIMIZER
        learning_rate = self.args.lr
        # model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())

        pa_others, pa_conv, pa_bias = [], [], []  # optimizer parameter groups
        for k, v in dict(self.model.named_parameters()).items():
            if '.bias' in k:
                pa_bias += [v]  # biases
            elif 'Conv2d.weight' in k:
                pa_conv += [v]  # apply weight_decay
            else:
                pa_others += [v]  # all else
        if opt_type == 'adam' or opt_type == 'Adam':
            self.optimizer = torch.optim.Adam(pa_others, lr=learning_rate)
        elif opt_type == 'sgd' or opt_type == 'SGD':
            self.optimizer = torch.optim.SGD(pa_others, lr=learning_rate, momentum=0.937, nesterov=True)
        else:
            self.cfg.logger.error('NO such a optimizer: ' + str(opt_type))
        self.optimizer.add_param_group({'params': pa_conv, 'weight_decay': 0.0005})  # add pa_conv with weight_decay
        self.optimizer.add_param_group({'params': pa_bias})  # add pa_bias (biases)
        del pa_others, pa_conv, pa_bias

        if self.optimizer_dict: self.optimizer.load_state_dict(self.optimizer_dict)
        if self.args.lr_continue:
            self.optimizer.param_groups[0]['lr'] = self.args.lr_continue
        self.learning_rate = self.optimizer.param_groups[0]['lr']
        if self.cfg.TRAIN.LR_SCHEDULE == 'cos':
            lf = lambda x: (((1 + math.cos(x * math.pi / self.cfg.TRAIN.EPOCH_SIZE)) / 2) ** 1.0) * 0.95 + 0.05  # ==0.05 cosine the last lr = 0.05xlr_start
            self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)
            # Plot lr schedule
            plot_lr = 0
            if plot_lr:
                y = []
                for _ in range(40, self.cfg.TRAIN.EPOCH_SIZE):
                    self.scheduler.step()
                    y.append(self.optimizer.param_groups[0]['lr'])
                plt.plot(y, '.-', label='LambdaLR')
                plt.xlabel('epoch')
                plt.ylabel('LR')
                plt.tight_layout()
                plt.savefig('LR.png', dpi=300)
        else:
            self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.cfg.TRAIN.STEP_LR, gamma=0.1)

        self.scheduler.last_epoch = self.epoch_last  # see link below

        self.optimizer.zero_grad()

    def _set_warmup_lr(self, optimizer):
        optimizer.param_groups[0]['lr'] = self.learning_rate / self.cfg.TRAIN.WARM_UP_STEP * (self.global_step + 1)
        return optimizer

    def _get_dataloader(self):
        """
        Get the self.model, learning_rate, epoch_last, train_set, test_set.
        :return: learning_rate, epoch_last, train_set, test_set.
        """
        self.DataFun = DataLoaderFactory(self.cfg, self.args)
        #  load the last data set
        train_set, test_set = _read_train_test_dataset(self.cfg)
        print('train set:', train_set[0], '\n', 'test set:', test_set[0])
        txt = 'train set:{}; test set:{}'.format(len(train_set), len(test_set))
        print(txt)
        self.cfg.logger.info(txt)
        self.trainDataloader, self.testDataloader = self.DataFun.make_dataset(train_set, test_set)

    def _calculate_loss(self, predict, dataset, **kwargs):
        total_loss = 0.
        loss_head_info = ''
        losses = self.lossfun.Loss_Call(predict, dataset, kwargs=kwargs)

        for k, v in losses.items():
            if k == 'metrics':
                continue
            total_loss += v
        try:
            loss_metrics = losses['metrics']
        except:
            loss_metrics = {}

        # add tensorboard writer.
        w_dict = {}
        if self.global_step % 1000 == 0:
            for k, v in loss_metrics.items():
                loss_head_info += ' %s: %.4f; ' % (k, v)
                w_dict['metrics/' + k] = v
            w_dict['epoch'] = self.global_step
            self.cfg.writer.tbX_write(w_dict=w_dict)

        self.cfg.logger.debug(loss_head_info)
        if torch.isnan(total_loss) or total_loss.item() == float("inf") or total_loss.item() == -float("inf"):
            self.cfg.logger.error("received an nan/inf loss:", dataset[-1])
            exit()
        return total_loss, loss_metrics

    def _save_checkpoint(self):
        _model = self.ema.ema if self.ema else self.model
        saved_dict = {'state_dict': _model.state_dict(),
                      'epoch': self.epoch,
                      'optimizer': self.optimizer.state_dict(),
                      'global_step': self.global_step}

        path_list = [str(self.epoch), 'now']
        for path_i in path_list:
            checkpoint_path = os.path.join(self.cfg.PATH.TMP_PATH, 'checkpoints/' + self.cfg.TRAIN.MODEL, path_i + '.pkl')
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(saved_dict, checkpoint_path)
            print('checkpoint is saved to %s', checkpoint_path)

    def _load_checkpoint(self, model, checkpoint, device, pre_trained=False):
        new_dic = OrderedDict()
        checkpoint = torch.load(checkpoint, map_location=device)
        state_dict = checkpoint['state_dict']
        for k, v in state_dict.items():
            if 'module.' == k[:8]:
                k = k.replace('module.', '')
            new_dic[k] = v
        model.load_state_dict(new_dic)
        if pre_trained:
            last_epoch = 0
            optimizer_dict = None
            global_step = 0
        else:
            last_epoch = checkpoint['epoch']
            optimizer_dict = checkpoint['optimizer']
            global_step = checkpoint['global_step']
        return model, last_epoch, optimizer_dict, global_step