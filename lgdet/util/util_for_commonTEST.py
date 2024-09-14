# from torchsummary import summary
# import torchvision.models as models
# model = models.resnet152()
# model = model.cuda()
# summary(model, input_size=(3,224,224), batch_size=-1, device='cuda')


a = set((1, 2, 3, 4))
b = {1, 2}
print(a, b)
print(a - b)
print(hex(16))


a = set()

a = set(((-1, 0, 0, 1), (-2, -1, 1, 2), (-2, 0, 0, 2)))
a.add((-1, 0, 0, 1))
print(a)
