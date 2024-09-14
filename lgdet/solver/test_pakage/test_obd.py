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
import torch


class Test_OBD(TestBase):
    def __init__(self, cfg, args, train):
        super(Test_OBD, self).__init__(cfg, args, train)
        self.parsepredict = ParsePredict(cfg)
        self.apolloclass2num = dict(zip(self.cfg.TRAIN.CLASSES, range(len(self.cfg.TRAIN.CLASSES))))
        self.ustrt = 0
        if self.ustrt:
            from torch2trt import TRTModule
            self.model_trt_2 = TRTModule()
            self.model_trt_2.load_state_dict(torch.load('others/model_compression/torch2tensorrt/tmp/yolov5_with_model.pth.onnx.statedict_trt'))

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
            if self.ustrt:
                predicts_trt = self.model_trt_2(inputs)
                dis = predicts_trt[0] - predicts[0]
                print('dis:', dis.max())
                predict_list = [predicts, predicts_trt]
            else:
                predict_list = [predicts]
            for predict in predict_list:
                labels_pres = self.parsepredict.parse_predict(predict)
                labels_pres = self.parsepredict.predict2labels(labels_pres, data_infos)
                batches = 1
                timer.time_end()
                print('a batch time is', timer.diff)
                for i in range(batches):
                    img_raw = [cv2.imread(data_infos[i]['img_path'])]
                    img_in = inputs[i]
                    _show_img(img_raw, labels_pres, img_in=img_in, pic_path=data_infos[i]['img_path'], cfg=self.cfg,
                              is_training=False, relative_labels=False)
