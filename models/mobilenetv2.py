import torch.nn as nn
import torch
import os
import math
from lib.utils.quantize_utils import QConv2d, QLinear

__all__ = ['MobileNetV2', 'mobilenetv2', 'qmobilenetv2']


def conv_bn(inp, oup, stride, conv_layer=nn.Conv2d, half_wave=True):
    if conv_layer == nn.Conv2d:
        return nn.Sequential(
            conv_layer(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
        )
    else:
        return nn.Sequential(
            conv_layer(inp, oup, 3, stride, 1, bias=False, half_wave=half_wave),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
        )


def conv_1x1_bn(inp, oup, conv_layer=nn.Conv2d, half_wave=True):
    if conv_layer == nn.Conv2d:
        return nn.Sequential(
            conv_layer(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
        )
    else:
        return nn.Sequential(
            conv_layer(inp, oup, 1, 1, 0, bias=False, half_wave=half_wave),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
        )


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, conv_layer=nn.Conv2d):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                conv_layer(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                conv_layer(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        elif conv_layer == nn.Conv2d:
            self.conv = nn.Sequential(
                # pw
                conv_layer(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                conv_layer(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                conv_layer(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                conv_layer(inp, hidden_dim, 1, 1, 0, bias=False, half_wave=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                conv_layer(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                conv_layer(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, input_size=224, width_mult=1., block=InvertedResidual, conv_layer=nn.Conv2d):
        super(MobileNetV2, self).__init__()
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2, conv_layer=conv_layer)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            if conv_layer is not nn.Conv2d:
                output_channel = make_divisible(c * width_mult)
            else:
                output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t, conv_layer=conv_layer))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t, conv_layer=conv_layer))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel, conv_layer=conv_layer, half_wave=False))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        if conv_layer == nn.Conv2d:
            self.classifier = nn.Linear(self.last_channel, num_classes)
        else:
            self.classifier = QLinear(self.last_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
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
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv2(pretrained=False, **kwargs):
    model = MobileNetV2(**kwargs)
    if pretrained:
        # Load pretrained model.
        path = 'pretrained/imagenet/mobilenetv2-150.pth.tar'
        print('==> load pretrained mobilenetv2 model..')
        assert os.path.isfile(path), 'Error: no checkpoint directory found!'
        ch = torch.load(path)
        ch = {n.replace('module.', ''): v for n, v in ch['state_dict'].items()}
        model.load_state_dict(ch, strict=False)
    return model


def qmobilenetv2(pretrained=False, num_classes=1000, **kwargs):
    # model = MobileNetV2(conv_layer=QConv2d, **kwargs)
    model = MobileNetV2(conv_layer=QConv2d, num_classes=1000, **kwargs)
    if pretrained:
        # Load pretrained model.
        path = 'pretrained/imagenet/mobilenetv2-150.pth.tar'
        print('==> load pretrained mobilenetv2 model..')
        assert os.path.isfile(path), 'Error: no checkpoint directory found!'
        ch = torch.load(path)
        ch = {n.replace('module.', ''): v for n, v in ch['state_dict'].items()}
        model.load_state_dict(ch, strict=False)
    return model


if __name__ == '__main__':
    # from ops.profile import profile
    net = mobilenetv2()
    flops, param = profile(net, input_size=(1, 3, 224, 224))
    print(flops/1e9, param/1e6)
