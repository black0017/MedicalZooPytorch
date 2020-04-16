from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1, padding=1):
    kernel_size = 3
    if dilation > 1:
        padding = find_padding(dilation, kernel_size)

    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=padding, dilation=dilation,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


def find_padding(dilation, kernel):
    """
    Dynamically computes padding to keep input conv size equal to the output
    for stride = 1
    :return:
    """
    return int(((kernel - 1) * (dilation - 1) + (kernel - 1)) / 2.0)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class TranspConvNet(nn.Module):
    """
    we replace the fully connected layer with a 8-branch decoder, where each branch consists of a 1x1x1 convolutional kernel
    and a corresponding up-sampling layer that scale the network output up to the original dimension.
    """

    def __init__(self, in_channels, classes):
        super().__init__()
        channels = 32
        transp_conv = nn.ConvTranspose3d(in_channels, channels, kernel_size=2, stride=2)
        batch_norm_1 = nn.BatchNorm3d(channels)
        relu_1 = nn.ReLU(inplace=True)

        self.transp_1 = nn.Sequential(transp_conv, batch_norm_1, relu_1)
        # TODO replace with conv3x3x3
        conv1 = nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        batch_norm_2 = nn.BatchNorm3d(channels)
        relu_2 = nn.ReLU(inplace=True)

        self.conv_1 = nn.Sequential(conv1, batch_norm_2, relu_2)
        # TODO replace with conv1x1x1
        self.conv_final = nn.Conv3d(channels, classes, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        x = self.transp_1(x)
        x = self.conv_1(x)
        y = self.conv_final(x)
        return y


class ResNet(nn.Module):

    def __init__(self, in_channels=3, classes=10,
                 block=BasicBlock,
                 layers=[1, 1, 1, 1],
                 block_inplanes=[64, 128, 256, 512],
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0
                 ):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(in_channels,
                               self.in_planes,
                               kernel_size=(7, 7, 7),
                               stride=(2, 2, 2),
                               padding=(3, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)

        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=1, dilation=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=1, dilation=4)

        self.segm = TranspConvNet(in_channels=512 * block.expansion, classes=classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride, dilation=dilation,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.segm(x)

        return x


def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]
    res_net_dict = {10: [1, 1, 1, 1], 18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3],
                    152: [3, 8, 36, 3], 200: [3, 24, 36, 3]}

    in_planes = [64, 128, 256, 512]

    if model_depth == 10:
        model = ResNet(block=BasicBlock, layers=res_net_dict[model_depth], block_inplanes=[64, 128, 256, 512], **kwargs)
    elif model_depth == 18:
        model = ResNet(block=BasicBlock, layers=res_net_dict[model_depth], block_inplanes=[64, 128, 256, 512], **kwargs)
    elif model_depth == 34:
        model = ResNet(block=BasicBlock, layers=res_net_dict[model_depth], block_inplanes=[64, 128, 256, 512], **kwargs)
    elif model_depth == 50:
        model = ResNet(block=Bottleneck, layers=res_net_dict[model_depth], block_inplanes=[64, 128, 256, 512], **kwargs)
    elif model_depth == 101:
        model = ResNet(block=Bottleneck, layers=res_net_dict[model_depth], block_inplanes=[64, 128, 256, 512], **kwargs)
    elif model_depth == 152:
        model = ResNet(block=Bottleneck, layers=[3, 8, 36, 3], block_inplanes=[64, 128, 256, 512], **kwargs)
    elif model_depth == 200:
        model = ResNet(block=Bottleneck, layers=[3, 24, 36, 3], block_inplanes=[64, 128, 256, 512], **kwargs)

    return model


depth = 50
model = generate_model(depth)
a = torch.rand(1, 3, 64, 64, 64)
y = model(a)
print(y.shape)
print('\n\n\n\n\n\n\n\n\n\n\n resnet {} ok \n\n\n\n\n\n\n\n\n\n\n').format(depth)
