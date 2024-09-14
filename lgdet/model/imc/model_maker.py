def get_network(net, num_class):
    """ return given network
    """

    if net == 'vgg16':
        from lgdet.model.imc.vgg import vgg16_bn
        net = vgg16_bn(num_class)
    elif net == 'vgg13':
        from lgdet.model.imc.vgg import vgg13_bn
        net = vgg13_bn(num_class)
    elif net == 'vgg11':
        from lgdet.model.imc.vgg import vgg11_bn
        net = vgg11_bn(num_class)
    elif net == 'vgg19':
        from lgdet.model.imc.vgg import vgg19_bn
        net = vgg19_bn(num_class)
    elif net == 'densenet121':
        from lgdet.model.imc.densenet import densenet121
        net = densenet121()
    elif net == 'densenet161':
        from lgdet.model.imc.densenet import densenet161
        net = densenet161()
    elif net == 'densenet169':
        from lgdet.model.imc.densenet import densenet169
        net = densenet169()
    elif net == 'densenet201':
        from lgdet.model.imc.densenet import densenet201
        net = densenet201()
    elif net == 'googlenet':
        from lgdet.model.imc.googlenet import googlenet
        net = googlenet()
    elif net == 'inceptionv3':
        from lgdet.model.imc.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif net == 'inceptionv4':
        from lgdet.model.imc.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif net == 'inceptionresnetv2':
        from lgdet.model.imc.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif net == 'xception':
        from lgdet.model.imc.xception import xception
        net = xception()
    elif net == 'resnet18':
        from lgdet.model.imc.resnet import resnet18
        net = resnet18()
    elif net == 'resnet34':
        from lgdet.model.imc.resnet import resnet34
        net = resnet34()
    elif net == 'resnet50':
        from lgdet.model.imc.resnet import resnet50
        net = resnet50()
    elif net == 'resnet101':
        from lgdet.model.imc.resnet import resnet101
        net = resnet101()
    elif net == 'resnet152':
        from lgdet.model.imc.resnet import resnet152
        net = resnet152()
    elif net == 'preactresnet18':
        from lgdet.model.imc.preactresnet import preactresnet18
        net = preactresnet18()
    elif net == 'preactresnet34':
        from lgdet.model.imc.preactresnet import preactresnet34
        net = preactresnet34()
    elif net == 'preactresnet50':
        from lgdet.model.imc.preactresnet import preactresnet50
        net = preactresnet50()
    elif net == 'preactresnet101':
        from lgdet.model.imc.preactresnet import preactresnet101
        net = preactresnet101()
    elif net == 'preactresnet152':
        from lgdet.model.imc.preactresnet import preactresnet152
        net = preactresnet152()
    elif net == 'resnext50':
        from lgdet.model.imc.resnext import resnext50
        net = resnext50()
    elif net == 'resnext101':
        from lgdet.model.imc.resnext import resnext101
        net = resnext101()
    elif net == 'resnext152':
        from lgdet.model.imc.resnext import resnext152
        net = resnext152()
    elif net == 'shufflenet':
        from lgdet.model.imc.shufflenet import shufflenet
        net = shufflenet()
    elif net == 'shufflenetv2':
        from lgdet.model.imc.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif net == 'squeezenet':
        from lgdet.model.imc.squeezenet import squeezenet
        net = squeezenet()
    elif net == 'mobilenet':
        from lgdet.model.imc.mobilenet import mobilenet
        net = mobilenet()
    elif net == 'mobilenetv2':
        from lgdet.model.imc.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif net == 'nasnet':
        from lgdet.model.imc.nasnet import nasnet
        net = nasnet()
    elif net == 'attention56':
        from lgdet.model.imc.attention import attention56
        net = attention56()
    elif net == 'attention92':
        from lgdet.model.imc.attention import attention92
        net = attention92()
    elif net == 'seresnet18':
        from lgdet.model.imc.senet import seresnet18
        net = seresnet18()
    elif net == 'seresnet34':
        from lgdet.model.imc.senet import seresnet34
        net = seresnet34()
    elif net == 'seresnet50':
        from lgdet.model.imc.senet import seresnet50
        net = seresnet50()
    elif net == 'seresnet101':
        from lgdet.model.imc.senet import seresnet101
        net = seresnet101()
    elif net == 'seresnet152':
        from lgdet.model.imc.senet import seresnet152
        net = seresnet152()
    elif net == 'wideresnet':
        from lgdet.model.imc.wideresidual import wideresnet
        net = wideresnet()
    elif net == 'stochasticdepth18':
        from lgdet.model.imc.stochasticdepth import stochastic_depth_resnet18
        net = stochastic_depth_resnet18()
    elif net == 'stochasticdepth34':
        from lgdet.model.imc.stochasticdepth import stochastic_depth_resnet34
        net = stochastic_depth_resnet34()
    elif net == 'stochasticdepth50':
        from lgdet.model.imc.stochasticdepth import stochastic_depth_resnet50
        net = stochastic_depth_resnet50()
    elif net == 'stochasticdepth101':
        from lgdet.model.imc.stochasticdepth import stochastic_depth_resnet101
        net = stochastic_depth_resnet101()

    else:
        print('the network name you have entered is not supported yet')

    return net


if __name__ == '__main__':
    imc_network = get_network('vgg16')
    a=0