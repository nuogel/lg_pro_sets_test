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
SCORES = LG_REGISTRY('score')
DATALOADERS = LG_REGISTRY('dataloader')


def build_from_cfg(Registry, type):
    obj_cls = Registry._module_dict.get(type, None)
    return obj_cls
