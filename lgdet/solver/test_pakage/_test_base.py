"""
Test.py is used for marking things with the weight file which we trained.

With a box outside of thing and with a label of what it is ,
and with it's score at the left top of the box.
"""
import os
import glob
from lgdet.solver.solver_base import BaseSolver


class TestBase(BaseSolver):
    def __init__(self, cfg, args, train):
        super().__init__(cfg, args, train)
        ...

    def test_backbone(self, DataSet):
        pass

    def test_camera(self, DataSet):
        pass

    def test_run(self, file_s):
        """
        Test images in the file_s.

        :param file_s:
        :return:
        """
        if file_s == 'one_name':
            self.cfg.TEST.ONE_TEST = 1
        elif file_s.strip().split('.')[-1] in ['mp4', 'avi']:
            self.test_camera(file_s)
        else:
            dataset = self.prase_file(file_s)
            DataSet = self.DataFun.make_dataset(train_dataset=None, test_dataset=dataset)[1]
            self.test_backbone(DataSet)

    def prase_file(self, file_s):
        dataset = []
        if file_s == 'one_name':
            dataset = self.cfg.TEST.ONE_NAME
        elif file_s == 'test_set':
            FILE_TXT = self.cfg.TRAIN.TRAIN_DATA_FROM_FILE[0]
            test_set = os.path.join('datasets/', self.cfg.BELONGS, FILE_TXT, FILE_TXT + '_test' + '.txt')
            lines = open(test_set, 'r', encoding='utf-8').readlines()
            for line in lines:
                tmp = line.strip().split("┣┫")
                dataset.append(tmp)

        elif os.path.isfile(file_s):
            if file_s.split('.')[1] == 'txt':  # .txt
                lines = open(file_s, 'r', encoding='utf-8').readlines()
                for line in lines:
                    tmp = line.strip().split("┣┫")
                    dataset.append(tmp)
            else:  # xx.jpg
                dataset.append([os.path.basename(file_s), file_s, file_s])

        elif os.path.isdir(file_s):
            files = glob.glob('{}/*.*'.format(file_s))
            for i, path in enumerate(files):
                # img = cv2.imread(path)
                name = os.path.basename(path)
                dataset.append([name, path, path])

        elif isinstance(list, file_s):
            dataset = file_s

        else:
            dataset = None
        return dataset
