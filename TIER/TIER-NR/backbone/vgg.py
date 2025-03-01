import torch
import torchvision.models as models

def vgg16_bn():
    vgg16_bn = models.vgg16_bn()
    pre = torch.load('./pretrained/vgg16_bn-6c64b313.pth')
    vgg16_bn.load_state_dict(pre)
    return vgg16_bn

def vgg16():
    vgg16 = models.vgg16()
    pre = torch.load('./pretrained/vgg16-397923af.pth')
    vgg16.load_state_dict(pre)
    return vgg16

def vgg19():
    vgg19 = models.vgg19()
    pre = torch.load('./pretrained/vgg19-dcbb9e9d.pth')
    vgg19.load_state_dict(pre)
    return vgg19

'''input = torch.randn((1,3,224,224))
model =vgg19()
fea = model.features(input)
print(fea.shape)'''



