import os
import torch
import torch.nn as nn
os.sys.path.insert(0, os.path.abspath("../.."))
from lib.utils.quantize_utils import QConv2d, QLinear
import math

__all__ = ['MobileNet', 'mobilenet', 'qmobilenet']


def conv_bn(inp, oup, stride, conv_layer=nn.Conv2d, half_wave=True):
    if conv_layer == nn.Conv2d:
        return nn.Sequential(
            conv_layer(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            # nn.ReLU6(inplace=True)
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            conv_layer(inp, oup, 3, stride, 1, bias=False, half_wave=half_wave),
            nn.BatchNorm2d(oup),
            # nn.ReLU6(inplace=True)
            nn.ReLU(inplace=True)
        )


def conv_dw(inp, oup, stride, conv_layer=nn.Conv2d):
    return nn.Sequential(
        conv_layer(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(inplace=True),

        conv_layer(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


class MobileNet(nn.Module):
    def __init__(self, num_classes=1000, conv_layer=nn.Conv2d, profile='normal', w_mul=1.):
        super(MobileNet, self).__init__()

        # original
        if profile == 'normal':
            in_planes = 32
            cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]
        else:
            raise NotImplementedError

        if conv_layer is not nn.Conv2d and w_mul ==2.:
            in_planes = 64
            cfg = [128, (256, 2), 256, (512, 2), 512, (1024, 2), 1024, 1024, 1024, 1024, 1024, (2048, 2), 2048]

        if conv_layer == nn.Conv2d:
            self.conv1 = conv_bn(3, in_planes, stride=2)
        else:
            self.conv1 = conv_bn(3, in_planes, stride=2, conv_layer=conv_layer, half_wave=False)

        self.features = self._make_layers(in_planes, cfg, conv_layer)

        if conv_layer == nn.Conv2d:
            self.classifier = nn.Sequential(
                nn.Linear(cfg[-1], num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                QLinear(cfg[-1], num_classes),
            )

        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = x.mean(3).mean(2)  # global average pooling

        x = self.classifier(x)
        return x

    def _make_layers(self, in_planes, cfg, conv_layer):
        layers = []
        for x in cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(conv_dw(in_planes, out_planes, stride, conv_layer=conv_layer))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or type(m) == QConv2d:
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear) or type(m) == QLinear:
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenet(pretrained=False, **kwargs):
    model = MobileNet(**kwargs)
    if pretrained:
        # Load pretrained model.
        raise NotImplementedError
    return model


def qmobilenet(pretrained=False, **kwargs):
    model = MobileNet(conv_layer=QConv2d, **kwargs)
    if pretrained:
        # Load pretrained model.
        raise NotImplementedError
    return model

