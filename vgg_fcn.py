# @Time    : 2024/06/01 21:09
# @Author  : Yang Xionghui
# @File    : vgg_fcn.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, num_classes),
        # )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(pretrained=False, batch_norm=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], batch_norm=batch_norm), **kwargs)
    if pretrained == True:
        if batch_norm == True:
            model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']), strict=False)
        else:
            model.load_state_dict(model_zoo.load_url(model_urls['vgg11']), strict=False)

    backbone = nn.ModuleList([])

    if batch_norm == True:
        backbone.append(model.features[0:14])
        backbone.append(model.features[14:21])
        backbone.append(model.features[21:28])
        backbone.append(model.features[28:])
    else:
        backbone.append(model.features[0:10])
        backbone.append(model.features[10:15])
        backbone.append(model.features[15:20])
        backbone.append(model.features[20:])

    return backbone


def vgg13(pretrained=False, batch_norm=False, **kwargs):
    """VGG 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B'], batch_norm=batch_norm), **kwargs)
    if pretrained == True:
        if batch_norm == True:
            model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']), strict=False)
        else:
            model.load_state_dict(model_zoo.load_url(model_urls['vgg13']), strict=False)

    backbone = nn.ModuleList([])

    if batch_norm == True:
        backbone.append(model.features[0:20])
        backbone.append(model.features[20:27])
        backbone.append(model.features[27:34])
        backbone.append(model.features[34:])
    else:
        backbone.append(model.features[0:14])
        backbone.append(model.features[14:19])
        backbone.append(model.features[19:24])
        backbone.append(model.features[24:])

    return backbone


def vgg16(pretrained=False, batch_norm=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=batch_norm), **kwargs)
    if pretrained == True:
        if batch_norm == True:
            model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']), strict=False)
        else:
            model.load_state_dict(model_zoo.load_url(model_urls['vgg16']), strict=False)

    backbone = nn.ModuleList([])
    
    if batch_norm == True:
        backbone.append(model.features[0:23])
        backbone.append(model.features[23:33])
        backbone.append(model.features[33:43])
        backbone.append(model.features[43:])
    else:
        backbone.append(model.features[0:16])
        backbone.append(model.features[16:23])
        backbone.append(model.features[23:30])
        backbone.append(model.features[30:])

    return backbone


def vgg19(pretrained=False, batch_norm=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], batch_norm=batch_norm), **kwargs)
    if pretrained == True:
        if batch_norm == True:
            model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']), strict=False)
        else:
            model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)

    backbone = nn.ModuleList([])
    
    if batch_norm == True:
        backbone.append(model.features[0:26])
        backbone.append(model.features[26:39])
        backbone.append(model.features[39:52])
        backbone.append(model.features[52:])
    else:
        backbone.append(model.features[0:18])
        backbone.append(model.features[18:27])
        backbone.append(model.features[27:36])
        backbone.append(model.features[36:])

    return backbone

class My_VGG(nn.Module):
    def __init__(self, pretrained=False, batch_norm=False, depth='16'):
        super(My_VGG, self).__init__()

        if depth == '11':
            self.backbone = vgg11(pretrained=pretrained, batch_norm=batch_norm)
        elif depth == '13':
            self.backbone = vgg13(pretrained=pretrained, batch_norm=batch_norm)
        elif depth == '16':
            self.backbone = vgg16(pretrained=pretrained, batch_norm=batch_norm)
        elif depth == '19':
            self.backbone = vgg19(pretrained=pretrained, batch_norm=batch_norm)
        
        print(pretrained,batch_norm)

    def forward(self, x):

        for i in range(4):
            x = self.backbone[i](x)
            print(x.shape)

        return x

if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224).cuda()
    model = My_VGG(pretrained = True, batch_norm = False, depth='19').cuda()
    print(model)
    y = model(x)