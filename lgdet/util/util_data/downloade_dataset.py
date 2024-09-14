import torchvision

traindata = torchvision.datasets.CIFAR10(root='/media/dell/data/cifar10', train=True, download=True)
testdata = torchvision.datasets.CIFAR10(root='/media/dell/data/cifar10', train=False, download=True)
a=0

