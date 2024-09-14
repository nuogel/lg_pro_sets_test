import torch.nn as nn


class LG_REGISTRY:
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def _register_module(self, module_class, module_name):
        if module_name is None:
            module_name = module_class.__name__
        self._module_dict[module_name] = module_class

    def registry(self, name=None):
        def _registry(cls):
            self._register_module(cls, name)
            return cls

        return _registry


MODELS = LG_REGISTRY('model')
LOSSES = LG_REGISTRY('loss')


@MODELS.registry()
class LG_MODEL1(nn.Module):
    def __init__(self, cfg):
        super(LG_MODEL1, self).__init__()  # 所以super().__init__()就是执行父类的构造函数，使得我们能够调用父类的属性。
        ...

    def forward(self):
        print('LG_MODEL1.forward()')


@MODELS.registry()
class LG_MODEL2:
    def __init__(self, cfg):
        ...

    def forward(self):
        print('LG_MODEL2.forward()')


@LOSSES.registry()
class LG_LOSS1:
    def __init__(self, cfg):
        ...

    def forward(self):
        print('LG_LOSS1.forward()')


def build_from_cfg(Registry, cfg):
    obj_cls = Registry._module_dict.get(cfg['type'], None)(cfg)
    return obj_cls


if __name__ == '__main__':
    model = {}
    loss = {}

    model['type'] = 'LG_MODEL2'
    loss['type'] = 'LG_LOSS1'

    model = build_from_cfg(MODELS, model)
    loss = build_from_cfg(LOSSES, loss)
    model.forward()
    loss.forward()
