"""
Test.py is used for marking things with the weight file which we trained.

With a box outside of thing and with a label of what it is ,
and with it's score at the left top of the box.
"""
import os
import torch
from lgdet.util.util_parse_SR_img import parse_Tensor_img
from lgdet.solver.test_pakage._test_base import TestBase

class Test_SRDN(TestBase):
    def __init__(self, cfg, args, train):
        super(Test_SRDN, self).__init__(cfg, args, train)
        # self.DataLoader = self.DataFun.DataLoaderDict[cfg.BELONGS](cfg)

    def test_backbone(self, DataSet):
        """Test."""
        loader = iter(DataSet)

        for i in range(DataSet.__len__()):
            test_data = next(loader)
            test_data = self.DataFun.to_devce(test_data)
            inputs, targets, data_infos = test_data
            predicts = self.model.forward(input_x=inputs, is_training=False)
            predicts = predicts.permute(0, 2, 3, 1)

            batches = inputs.shape[0]
            save_paths = []
            if self.cfg.TEST.SAVE_LABELS:
                for i in range(batches):
                    data_info = data_infos[i]
                    os.makedirs(self.cfg.PATH.GENERATE_LABEL_SAVE_PATH, exist_ok=True)
                    os.makedirs(os.path.join(self.cfg.PATH.GENERATE_LABEL_SAVE_PATH, self.cfg.TRAIN.MODEL),
                                exist_ok=True)
                    save_paths.append(
                        os.path.join(self.cfg.PATH.GENERATE_LABEL_SAVE_PATH, self.cfg.TRAIN.MODEL + '/' + data_info[0]))

            predict_size = (predicts.shape[1], predicts.shape[2])
            inputs = torch.nn.functional.interpolate(inputs, size=predict_size)
            inputs = inputs.permute(0, 2, 3, 1)
            targets = targets.permute(0, 2, 3, 1)
            inputs_join_predicts = 1
            if inputs_join_predicts:
                try:
                    img_cat = torch.cat([inputs, predicts, targets], dim=1)
                except:
                    img_cat = torch.cat([inputs, predicts], dim=1)
            else:
                img_cat = predicts

            parse_Tensor_img(img_cat, pixcels_norm=self.cfg.TRAIN.PIXCELS_NORM, save_paths=save_paths,
                             show_time=self.cfg.TEST.SHOW_EVAL_TIME)
