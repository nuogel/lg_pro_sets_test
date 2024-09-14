from torch.utils.data import DataLoader


class BaseLoader(DataLoader):
    def __init__(self, cfg, dataset, is_training):
        super(BaseLoader, self).__init__(object)
        self.cfg = cfg
        self.is_training = is_training
        self.one_test = cfg.TEST.ONE_TEST
        self.one_name = cfg.TEST.ONE_NAME
        self.datapath = cfg.PATH.INPUT_PATH
        self.dataset_infos = dataset

    def __len__(self):
        if self.one_test:
            if self.is_training:
                length = int(self.cfg.TEST.ONE_TEST_TRAIN_STEP) * self.cfg.TRAIN.BATCH_SIZE
            else:
                length = self.cfg.TRAIN.BATCH_SIZE
        else:
            length = len(self.dataset_infos)
        return length
