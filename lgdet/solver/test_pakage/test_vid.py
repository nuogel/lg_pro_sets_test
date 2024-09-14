"""
Test.py is used for marking things with the weight file which we trained.

With a box outside of thing and with a label of what it is ,
and with it's score at the left top of the box.
"""
import cv2
from lgdet.postprocess.parse_factory import ParsePredict
from lgdet.util.util_show_img import _show_img
from lgdet.util.util_time_stamp import Time
from lgdet.solver.test_pakage._test_base import TestBase


class Test_VID(TestBase):
    def __init__(self, cfg, args, train):
        super(Test_VID, self).__init__(cfg, args, train)
        self.parsepredict = ParsePredict(cfg)
        self.apolloclass2num = dict(zip(self.cfg.TRAIN.CLASSES, range(len(self.cfg.TRAIN.CLASSES))))

    def test_backbone(self, DataSet):
        """Test."""
        loader = iter(DataSet)
        timer = Time()
        for i in range(DataSet.__len__()):
            test_data = next(loader)
            timer.time_start()
            test_data = self.DataFun.to_devce(test_data)
            inputs, targets, data_infos = test_data
            predicts = self.model.forward(input_x=inputs, is_training=False)
            labels_pres = self.parsepredict.parse_predict(predicts)
            labels_pres = self.parsepredict.predict2labels(labels_pres, data_infos)
            batches = 1
            timer.time_end()
            print('a batch time is', timer.diff)
            for i in range(batches):
                img_raw = [cv2.imread(data_infos[i]['img_path'])]
                img_in = inputs[i]
                _show_img(img_raw, labels_pres, img_in=img_in, pic_path=data_infos[i]['img_path'], cfg=self.cfg,
                          is_training=False, relative_labels=False)
