"""Train the model.
writer:luogeng 2017.02
Train.py is used for training the net to mark things with a box outside of
them, and with a label of what it is, and with it's score at the left top of
the box.

At the end, we will get the weight file of the net.

"""
import tqdm
from .solver_base import BaseSolver
import torch
from torch.cuda import amp


class Solver(BaseSolver):
    def __init__(self, cfg, args, train=True):
        super().__init__(cfg, args, train)
        self._get_optimizer()
        self._get_score()
        self._get_lossfun()
        self.metrics_ave = {}
        self.amp = amp
        self.scaler = amp.GradScaler()

    def train(self):
        """Train the network.
        """
        for epoch in range(self.epoch_last, self.cfg.TRAIN.EPOCH_SIZE):
            self.epoch = epoch
            if not self.cfg.TEST.TEST_ONLY and not self.args.test_only:
                self._train_an_epoch(epoch)
            self._validate_an_epoch(epoch)

    def _train_an_epoch(self, epoch):
        self.model.train()
        # count the step time, total time...
        headinfos = ('\n[train] %8s|%7s|%7s|%9s|' + '%6s|' * 3) % ('model_n', 'epoch', 'g_step', 'l_rate', 'step_l', 'aver_l', 'others')
        print(headinfos)

        Pbar = tqdm.tqdm(self.trainDataloader)
        # time5 = time.time()
        for step, train_data in enumerate(Pbar):
            if self.global_step < self.cfg.TRAIN.WARM_UP_STEP:
                self._set_warmup_lr()
            train_data = self.DataFun.to_devce(train_data)
            # forward process
            if self.cfg.TRAIN.AMP:
                useamp = True
            else:
                useamp = False
            with self.amp.autocast(enabled=useamp):
                predict = self.model(input_x=train_data[0], input_y=train_data[1], input_data=train_data, is_training=False)
                # calculate the total loss
                self.global_step += 1
                total_loss, train_info = self._calculate_loss(predict, train_data,
                                                              losstype=self.cfg.TRAIN.LOSSTYPE,
                                                              global_step=self.global_step,
                                                              len_batch=len(self.trainDataloader),
                                                              step=step,
                                                              epoch=epoch)
                # except:
                #     print('continue')
                #     continue
                if total_loss is None:
                    continue
            # backward process
            if self.cfg.TRAIN.AMP:
                self.scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            if self.global_step % self.cfg.TRAIN.SAVE_STEP == 0:
                self._save_checkpoint()
            if self.global_step % self.cfg.TRAIN.BATCH_BACKWARD_SIZE == 0:
                if self.cfg.TRAIN.AMP:
                    self.scaler.step(self.optimizer)  # self.optimizer.step()
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                if self.cfg.TRAIN.EMA:
                    self.ema.update(self.model)
            Pbar.set_description(train_info)
        if self.cfg.TRAIN.LR_SCHEDULE == 'reduce':
            self.scheduler.step(self.epoch_losses / (step + 1))
        else:
            self.scheduler.step()
        if self.cfg.TRAIN.EMA:
            self.ema.update_attr(self.model)
        self._save_checkpoint()
        self.cfg.logger.info(train_info)

    def _validate_an_epoch(self, epoch):
        if ((epoch + 1) % self.cfg.TRAIN.EVALUATE_STEP == 0 and self.global_step > self.cfg.TRAIN.WARM_UP_STEP) or self.cfg.TEST.ONE_TEST:
            with torch.no_grad():
                if not self.cfg.TEST.ONE_TEST:
                    self.model.eval()
                self.cfg.logger.debug('[evaluate] model:%s, evaluating ...', self.cfg.TRAIN.MODEL)
                self.score.init_parameters()
                Pbar = tqdm.tqdm(self.testDataloader)
                for step, train_data in enumerate(Pbar):
                    if step >= len(self.trainDataloader):
                        break
                    test_data = self.DataFun.to_devce(train_data)
                    if test_data[0] is None: continue
                    predict = self.model(input_x=test_data[0], input_y=test_data[1], input_data=test_data, is_training=False)
                    self.score.cal_score(predict, test_data)
                    Pbar.set_description('[valid]')

                main_score, item_score = self.score.score_out()
                item_score['main_score'] = main_score
                w_dict = {'epoch': epoch, 'scores': item_score}
                self.cfg.writer.tbX_write(w_dict)
                self.cfg.logger.info('[EVALUATE] Summary: Epoch: %s, total_score: %s, other_score: %s', epoch, str(main_score), str(item_score))
                if self.cfg.TEST.TEST_ONLY:
                    exit()
