import yaml
import os
import munch


# class AttrDict(dict):
#     """
#     If name in dict,then get the name of the dict.
#     """
#
#     def __getattr__(self, name):
#         if name in self.__dict__:
#             return self.__dict__[name]
#         elif name in self:
#             return self[name]
#         else:
#             raise AttributeError(name)
#
#     def __setattr__(self, name, value):
#         if name in self.__dict__:
#             self.__dict__[name] = value
#         else:
#             self[name] = value


def parse_yaml(agrs):
    """
    Convert the dict to be [cfg.Name...]
    :param ymlfilename:
    :return:
    e.g :
        yaml_cfg['PATH']['TMP_DIR']
    """
    yml_path = os.path.join(os.path.dirname(__file__), '../../cfg/{}.yml'.format(agrs.type))
    f = open(yml_path, 'r', encoding='UTF-8')
    yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = munch.Munch(yaml_cfg)
    # cfg = AttrDict(yaml_cfg)
    cfg.PATH = munch.Munch(cfg.PATH)
    cfg.TRAIN = munch.Munch(cfg.TRAIN)
    cfg.TEST = munch.Munch(cfg.TEST)
    return cfg
