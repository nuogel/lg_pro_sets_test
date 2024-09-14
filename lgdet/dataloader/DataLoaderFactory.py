import torch
from torch.utils.data import DataLoader
from ..registry import DATALOADERS, build_from_cfg
from .group_sampler import GroupSampler
from lgdet.util.util_auto_kmeans import auto_kmeans_anchors


class DataLoaderFactory:
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args

    def make_dataset(self, train_dataset=None, test_dataset=None, cfg=None):
        shuffle = True
        trainLoader, testLoader = None, None
        train_data, test_data = None, None
        if train_dataset:
            train_data = build_from_cfg(DATALOADERS, self.cfg.BELONGS.upper() + '_Loader')(self.cfg, train_dataset, is_training=True)
            # sampler = GroupSampler(train_data, self.cfg.TRAIN.BATCH_SIZE) if shuffle is False else None
            if self.cfg.BELONGS == 'vid': shuffle = False
            trainLoader = DataLoader(dataset=train_data,
                                     batch_size=self.cfg.TRAIN.BATCH_SIZE,
                                     sampler=None,
                                     shuffle=shuffle,
                                     pin_memory=False,  # 这样将内存的Tensor转义到GPU的显存就会更快一些
                                     num_workers=self.args.number_works,
                                     collate_fn=train_data.collate_fun,
                                     )

        if test_dataset:
            # test_data = get_loader_class(self.cfg.BELONGS)(self.cfg)
            test_data = build_from_cfg(DATALOADERS, self.cfg.BELONGS.upper() + '_Loader')(self.cfg, test_dataset, is_training=False)
            testLoader = DataLoader(dataset=test_data,
                                    batch_size=self.cfg.TRAIN.BATCH_SIZE,
                                    num_workers=self.args.number_works,
                                    pin_memory=False,
                                    collate_fn=test_data.collate_fun,
                                    shuffle=False)

        if self.cfg.TRAIN.AUTO_ANCHORS and train_data != None and not self.cfg.TEST.ONE_TEST:
            anchors = auto_kmeans_anchors([train_data, test_data], self.cfg)
            cfg.TRAIN.ANCHORS = anchors
            self.cfg.logger.info('use auto anchors:'+ str(anchors))
        return trainLoader, testLoader

    def to_devce(self, data):
        for i, da in enumerate(data):
            if isinstance(da, (list, tuple)):
                for j, da_j in enumerate(da):
                    try:
                        data[i][j] = data[i][j].to(self.cfg.TRAIN.DEVICE)
                    except:
                        pass
            else:
                try:
                    data[i] = data[i].to(self.cfg.TRAIN.DEVICE)
                except:
                    pass
        return data
