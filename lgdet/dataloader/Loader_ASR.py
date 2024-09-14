from .loader_base import BaseLoader
from lgdet.util.util_audio.util_audio import Audio
from ..registry import DATALOADERS


@DATALOADERS.registry()
class ASR_Loader(BaseLoader):
    def __init__(self, cfg, dataset, is_training):
        super(ASR_Loader, self).__init__(cfg, dataset, is_training)
        self.audio = Audio(cfg.TRAIN)

    def __getitem__(self, index):
        if self.one_test:
            data_info = self.dataset_infos[0]
        else:
            data_info = self.dataset_infos[index]




    def _pinying2number(self, pinying):
        labnumber = []
        for i in pinying:
            labnumber.append(self.list_symbol.index(i))
        return labnumber

    def _number2pinying(self, num):
        pingyin = []
        for i in num:
            # print(i)
            pingyin.append(self.list_symbol[i])
        return pingyin
