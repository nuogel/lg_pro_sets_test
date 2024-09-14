"""
Test.py is used for marking things with the weight file which we trained.

With a box outside of thing and with a label of what it is ,
and with it's score at the left top of the box.
"""
from lgdet.solver.test_pakage._test_base import TestBase



class Test_ASR(TestBase):
    def __init__(self, cfg, args, train):
        super(Test_ASR, self).__init__(cfg, args, train)
        self.DataLoader = self.DataFun.DataLoaderDict

    def test_backbone(self, wav_path):
        """Test."""
        # prepare paramertas
        self.cfg.TRAIN.BATCH_SIZE = 1
        test_data = self.DataLoader.get_one_data_for_test(wav_path)
        predict = self.model.forward(test_data, is_training=False)
        for k, v in predict.items():
            print('pre:', k, self.DataLoader._number2pinying(v[:-1]))