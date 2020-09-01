import torch
from torch.utils.data import DataLoader
from ..registry import DATALOADERS, build_from_cfg


class DataLoaderFactory:
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args

    def make_dataset(self, train_dataset=None, test_dataset=None):
        shuffle = True
        trainLoader, testLoader = None, None

        if train_dataset:
            # train_data = get_loader_class(self.cfg.BELONGS)(self.cfg)
            train_data = build_from_cfg(DATALOADERS, self.cfg.BELONGS + '_Loader')(self.cfg)
            train_data._add_dataset(train_dataset, is_training=True)
            if self.cfg.BELONGS == 'VID': shuffle = False
            trainLoader = DataLoader(dataset=train_data,
                                     batch_size=self.cfg.TRAIN.BATCH_SIZE,
                                     num_workers=self.args.number_works,
                                     collate_fn=train_data.collate_fun,
                                     shuffle=shuffle)

        if test_dataset:
            # test_data = get_loader_class(self.cfg.BELONGS)(self.cfg)
            test_data = build_from_cfg(DATALOADERS, self.cfg.BELONGS + '_Loader')(self.cfg)
            test_data._add_dataset(test_dataset, is_training=False)
            testLoader = DataLoader(dataset=test_data,
                                    batch_size=self.cfg.TRAIN.BATCH_SIZE,
                                    num_workers=self.args.number_works,
                                    collate_fn=test_data.collate_fun,
                                    shuffle=False)

        return trainLoader, testLoader

    def to_devce(self, data):
        datas = []
        for i, da in enumerate(data):
            try:
                datas.append(da.to(self.cfg.TRAIN.DEVICE, non_blocking=True))
            except:
                datas.append(da)
        return datas
