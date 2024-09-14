"""Parse the predictions."""
from lgdet.util.util_lg_transformer import LgTransformer

from lgdet.postprocess.parse_yolo import ParsePredict_yolo
from lgdet.postprocess.parse_yolox import ParsePredict_yolox
from lgdet.postprocess.parse_multibox import ParsePredict_multibox
from lgdet.postprocess.parse_refinedet import ParsePredict_refinedet
from lgdet.postprocess.parse_fcos import ParsePredict_fcos


class ParsePredict:
    # TODO: 分解parseprdict.
    def __init__(self, cfg):
        self.cfg = cfg
        self.parse_dict = {
            'yolo': ParsePredict_yolo,
            'yolox': ParsePredict_yolox,
            # 'yolov2': ParsePredict_yolo,
            # 'yolov3': ParsePredict_yolo,
            # 'yolov3_tiny': ParsePredict_yolo,
            # 'yolov3_tiny_o': ParsePredict_yolo,
            # 'yolov3_tiny_mobilenet': ParsePredict_yolo,
            # 'yolov3_tiny_squeezenet': ParsePredict_yolo,
            # 'yolov3_tiny_shufflenet': ParsePredict_yolo,
            # 'yolov2_fgfa': ParsePredict_yolo,
            # 'yolov5': ParsePredict_yolo,
            # 'pvt_yolov5': ParsePredict_yolo,
            # 'yolonano': ParsePredict_yolo,
            'fcos': ParsePredict_fcos,
            'refinedet': ParsePredict_refinedet,
            'efficientdet': ParsePredict_multibox,
            'ssdvgg': ParsePredict_multibox,
            'retinanet': ParsePredict_multibox,
            'pvt_retinanet': ParsePredict_multibox,
            'lrf300': ParsePredict_multibox,
            'lrf512': ParsePredict_multibox,
        }

        self.transformer = LgTransformer(self.cfg)
        modelname = self.cfg.TRAIN.MODEL
        for name in ['yolox', 'yolo']:
            if name in modelname:
                modelname = name
                break
        self.parser = self.parse_dict[modelname](self.cfg)

    def parse_predict(self, f_maps):
        labels_predict = self.parser.parse_predict(f_maps)
        return labels_predict

    def predict2labels(self, labels_predict, data_infos):
        labels_predict = self.transformer.decode_pad2size(labels_predict, data_infos)  # absolute labels
        return labels_predict
