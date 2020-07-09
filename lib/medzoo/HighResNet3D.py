import torch
import torch.nn as nn
from lib.medzoo.BaseModelClass import BaseModel

"""
Implementation based on the paper:
https://arxiv.org/pdf/1707.01992.pdf
"""


class ConvInit(nn.Module):
    def __init__(self, in_channels):
        super(ConvInit, self).__init__()
        self.num_features = 16
        self.in_channels = in_channels

        self.conv1 = nn.Conv3d(self.in_channels, self.num_features, kernel_size=3, padding=1)
        bn1 = torch.nn.BatchNorm3d(self.num_features)
        relu1 = nn.ReLU()

        self.norm = nn.Sequential(bn1, relu1)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.norm(y1)

        return y1, y2


class ConvRed(nn.Module):
    def __init__(self, in_channels):
        super(ConvRed, self).__init__()
        self.num_features = 16
        self.in_channels = in_channels

        bn1 = torch.nn.BatchNorm3d(self.in_channels)
        relu1 = nn.ReLU()
        conv1 = nn.Conv3d(self.in_channels, self.num_features, kernel_size=3, padding=1)
        self.conv_red = nn.Sequential(bn1, relu1, conv1)

    def forward(self, x):
        return self.conv_red(x)


class DilatedConv2(nn.Module):
    def __init__(self, in_channels):
        super(DilatedConv2, self).__init__()
        self.num_features = 32
        self.in_channels = in_channels
        bn1 = torch.nn.BatchNorm3d(self.in_channels)
        relu1 = nn.ReLU()
        conv1 = nn.Conv3d(self.in_channels, self.num_features, kernel_size=3, padding=2, dilation=2)

        self.conv_dil = nn.Sequential(bn1, relu1, conv1)

    def forward(self, x):
        return self.conv_dil(x)


class DilatedConv4(nn.Module):
    def __init__(self, in_channels):
        super(DilatedConv4, self).__init__()
        self.num_features = 64
        self.in_channels = in_channels

        bn1 = torch.nn.BatchNorm3d(self.in_channels)
        relu1 = nn.ReLU()
        conv1 = nn.Conv3d(self.in_channels, self.num_features, kernel_size=3, padding=4, dilation=4)

        self.conv_dil = nn.Sequential(bn1, relu1, conv1)

    def forward(self, x):
        return self.conv_dil(x)


class Conv1x1x1(nn.Module):
    def __init__(self, in_channels, classes):
        super(Conv1x1x1, self).__init__()
        self.num_features = classes
        self.in_channels = in_channels

        bn1 = torch.nn.BatchNorm3d(self.in_channels)
        relu1 = nn.ReLU()
        conv1 = nn.Conv3d(self.in_channels, self.num_features, kernel_size=1)

        self.conv_dil = nn.Sequential(bn1, relu1, conv1)

    def forward(self, x):
        return self.conv_dil(x)


class HighResNet3D(BaseModel):
    def __init__(self, in_channels=1, classes=4, shortcut_type="A", dropout_layer=True):
        super(HighResNet3D, self).__init__()
        self.in_channels = in_channels
        self.shortcut_type = shortcut_type
        self.classes = classes
        self.init_channels = 16
        self.red_channels = 16
        self.dil2_channels = 32
        self.dil4_channels = 64
        self.conv_out_channels = 80

        if self.shortcut_type == "B":
            self.res_pad_1 = Conv1x1x1(self.red_channels, self.dil2_channels)
            self.res_pad_2 = Conv1x1x1(self.dil2_channels, self.dil4_channels)

        self.conv_init = ConvInit(in_channels)

        self.red_blocks1 = self.create_red(self.init_channels)
        self.red_blocks2 = self.create_red(self.red_channels)
        self.red_blocks3 = self.create_red(self.red_channels)

        self.dil2block1 = self.create_dil2(self.red_channels)
        self.dil2block2 = self.create_dil2(self.dil2_channels)
        self.dil2block3 = self.create_dil2(self.dil2_channels)

        self.dil4block1 = self.create_dil4(self.dil2_channels)
        self.dil4block2 = self.create_dil4(self.dil4_channels)
        self.dil4block3 = self.create_dil4(self.dil4_channels)

        if dropout_layer:
            conv_out = nn.Conv3d(self.dil4_channels, self.conv_out_channels, kernel_size=1)
            drop3d = nn.Dropout3d()
            conv1x1x1 = Conv1x1x1(self.conv_out_channels, self.classes)
            self.conv_out = nn.Sequential(conv_out, drop3d, conv1x1x1)
        else:
            self.conv_out = Conv1x1x1(self.dil4_channels, self.classes)

    def shortcut_pad(self, x, desired_channels):
        if self.shortcut_type == 'A':
            batch_size, channels, dim0, dim1, dim2 = x.shape
            extra_channels = desired_channels - channels
            zero_channels = int(extra_channels / 2)
            zeros_half = x.new_zeros(batch_size, zero_channels, dim0, dim1, dim2)
            y = torch.cat((zeros_half, x, zeros_half), dim=1)
        elif self.shortcut_type == 'B':
            if desired_channels == self.dil2_channels:
                y = self.res_pad_1(x)
            elif desired_channels == self.dil4_channels:
                y = self.res_pad_2(x)
        return y

    def create_red(self, in_channels):
        conv_red_1 = ConvRed(in_channels)
        conv_red_2 = ConvRed(self.red_channels)
        return nn.Sequential(conv_red_1, conv_red_2)

    def create_dil2(self, in_channels):
        conv_dil2_1 = DilatedConv2(in_channels)
        conv_dil2_2 = DilatedConv2(self.dil2_channels)
        return nn.Sequential(conv_dil2_1, conv_dil2_2)

    def create_dil4(self, in_channels):
        conv_dil4_1 = DilatedConv4(in_channels)
        conv_dil4_2 = DilatedConv4(self.dil4_channels)
        return nn.Sequential(conv_dil4_1, conv_dil4_2)

    def red_forward(self, x):
        x, x_res = self.conv_init(x)
        x_red_1 = self.red_blocks1(x)
        x_red_2 = self.red_blocks2(x_red_1 + x_res)
        x_red_3 = self.red_blocks3(x_red_2 + x_red_1)
        return x_red_3, x_red_2

    def dilation2(self, x_red_3, x_red_2):
        x_dil2_1 = self.dil2block1(x_red_3 + x_red_2)
        # print(x_dil2_1.shape ,x_red_3.shape )

        x_red_padded = self.shortcut_pad(x_red_3, self.dil2_channels)

        x_dil2_2 = self.dil2block2(x_dil2_1 + x_red_padded)
        x_dil2_3 = self.dil2block3(x_dil2_2 + x_dil2_1)
        return x_dil2_3, x_dil2_2

    def dilation4(self, x_dil2_3, x_dil2_2):
        x_dil4_1 = self.dil4block1(x_dil2_3 + x_dil2_2)
        x_dil2_padded = self.shortcut_pad(x_dil2_3, self.dil4_channels)
        x_dil4_2 = self.dil4block2(x_dil4_1 + x_dil2_padded)
        x_dil4_3 = self.dil4block3(x_dil4_2 + x_dil4_1)
        return x_dil4_3 + x_dil4_2

    def forward(self, x):
        x_red_3, x_red_2 = self.red_forward(x)
        x_dil2_3, x_dil2_2 = self.dilation2(x_red_3, x_red_2)
        x_dil4 = self.dilation4(x_dil2_3, x_dil2_2)
        y = self.conv_out(x_dil4)
        return y

    def test(self):
        x = torch.rand(1, self.in_channels, 32, 32, 32)
        pred = self.forward(x)
        target = torch.rand(1, self.classes, 32, 32, 32)
        assert target.shape == pred.shape
        print('High3DResnet ok!')


def test_all_modules():
    a = torch.rand(1, 16, 32, 32, 32)
    m1 = ConvInit(in_channels=16)
    y, _ = m1(a)
    assert y.shape == a.shape, print(y.shape)
    print("ConvInit OK")

    m2 = ConvRed(in_channels=16)
    y = m2(a)
    assert y.shape == a.shape, print(y.shape)
    print("ConvRed OK")

    a = torch.rand(1, 32, 32, 32, 32)
    m3 = DilatedConv2(in_channels=32)
    y = m3(a)
    assert y.shape == a.shape, print(y.shape)
    print("DilatedConv2 OK")

    a = torch.rand(1, 64, 32, 32, 32)
    m4 = DilatedConv4(in_channels=64)
    y = m4(a)
    assert y.shape == a.shape, print(y.shape)
    print("DilatedConv4 OK")

    m4 = Conv1x1x1(in_channels=64, classes=4)
    y = m4(a)
    print(y.shape)

# test_all_modules()

#model = HighResNet3D(in_channels=1, classes=4)
#model.test()
