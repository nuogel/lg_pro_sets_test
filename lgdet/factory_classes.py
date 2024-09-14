def get_score_class(belongs):
    if belongs == 'VID': belongs = 'obd'
    class_name = 'Score_' + str(belongs).upper()
    model_file = __import__('lgdet.score.' + class_name, fromlist=[class_name])
    model_class = _get_sub_model(model_file, 'Score')
    return model_class


# def get_model_class(belongs, modelname):
#     belongs_uper = str(belongs).upper()
#     modelname_uper = str(modelname).upper()
#     model_file = belongs_uper[0] + belongs_uper[1:].lower() + 'Model_' + modelname_uper
#     model_class = _get_sub_model(__import__('lgdet.model.' + model_file, fromlist=[modelname_uper]), modelname_uper)
#     return model_class


# def get_loader_class(belongs):
#     class_name = 'Loader_' + str(belongs).upper()
#     model_file = __import__('lgdet.dataloader.' + class_name, fromlist=[class_name])
#     model_class = _get_sub_model(model_file, 'Loader')
#     return model_class


def get_loss_class(belongs, modelname):
    if belongs == 'SRDN':
        from lgdet.loss.loss_srdn import SRDNLOSS
        return SRDNLOSS
    elif belongs == 'imc':
        from lgdet.loss.loss_imc import IMCLoss
        return IMCLoss
    elif belongs == 'obd':
        if modelname == 'yolov3':
            from lgdet.loss.loss_yolo_v3 import YoloLoss
            return YoloLoss
        elif 'yolov5' in modelname:
            from lgdet.loss.loss_yolo_v5 import YoloLoss
            return YoloLoss
        elif 'yolox' in modelname:
            from lgdet.loss.loss_yolox import YoloxLoss
            return YoloxLoss

        elif modelname[:4] == 'yolo':
            from lgdet.loss.loss_yolo import YoloLoss
            return YoloLoss
        elif modelname in ['ssdvgg', 'lrf300', 'lrf512']:
            from lgdet.loss.loss_multibox import MULTIBOXLOSS
            return MULTIBOXLOSS
        else:
            from lgdet.loss.loss_fcos import FCOSLOSS
            from lgdet.loss.loss_ctc import RnnLoss
            from lgdet.loss.loss_seq2seq import SEQ2SEQLOSS
            from lgdet.loss.loss_refinedet import REFINEDETLOSS
            from lgdet.loss.loss_pan import PANLoss
            from lgdet.loss.loss_flow import FlowLoss
            from lgdet.loss.loss_tacotron import TACOTRONLOSS
            from lgdet.loss.loss_retinanet import RETINANETLOSS
            loss_dict = {
                'fcos': FCOSLOSS,
                'refinedet': REFINEDETLOSS,
                'retinanet': RETINANETLOSS,
                'pvt_retinanet': RETINANETLOSS,
                'efficientdet': RETINANETLOSS,
                # ASR
                'rnn': RnnLoss,
                'ctc': RnnLoss,
                'seq2seq': SEQ2SEQLOSS,
                'PAN': PANLoss,
                'flow_fgfa': FlowLoss,
                # TTS
                'tacotron2': TACOTRONLOSS,
            }

            return loss_dict[modelname]


def _get_sub_model(model_file, class_name):
    if hasattr(model_file, class_name):
        class_model = getattr(model_file, class_name)
    else:
        class_model = None
        ImportError('no such a model')
    return class_model
