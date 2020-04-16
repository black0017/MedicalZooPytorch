from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.medzoo.BaseModelClass import BaseModel

"""
Original paper here: https://arxiv.org/abs/1904.00625
Implementation is strongly and modified from here: https://github.com/kenshohara/3D-ResNets-PyTorch

Network architecture, taken from paper:
We adopt the ResNet family (layers with 10, 18, 34, 50, 101, 152, and 200) and pre-activationResNet-200 
architecture as the backbone of Med3D networks.
To enable the network to train with 3D medical data,
we modify the backbone network as follows:
1) we change the channel number of the first convolution layer from 3 to 1 due to the single channel volume input
2) we replace all 2D convolution kernels with the 3D version

3) we set the stride of the convolution kernels in blocks 3 and 4 equal to 1 to avoid down-sampling the feature maps

4) we use dilated convolutional layers with rate r= 2 as suggested in [deep-lab] for the following layers for the same purpose

5) we replace the fully connected layer with a 8-branch decoder, where each branch consists of a 1x1x1 convolutional kernel
 and a corresponding up-sampling layer that scale the network output up to the original dimension.

we optimize network parameters using the cross-entropy loss with the standard SGD method,
where the learning rate is set to 0.1, momentum set to 0.9 and weight decay set to 0.001.


o = output, p = padding, k = kernel_size, s = stride, d = dilation
For transpose-conv:
o = (i -1)*s - 2*p + k + output_padding
For conv layers :
o = [i + 2*p - k - (k-1)*(d-1)]/s + 1
"""


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1,padding=1):
    kernel_size =3
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
    for stride =1
    :return:
    """
    return int(((kernel - 1) * (dilation - 1) + (kernel - 1)) / 2.0)


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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        #print('s,d=', stride, dilation)

        self.conv1 = conv3x3x3(in_planes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, stride=1, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

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
        self.conv2 = conv3x3x3(planes, planes, stride=stride, dilation=dilation)
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


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=2,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=10):
        super().__init__()

        self.input_channels = input_channels
        self.n_classes = n_classes

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        conv1 = nn.Conv3d(input_channels,
                          self.in_planes,
                          kernel_size=(conv1_t_size, 7, 7),
                          stride=(conv1_t_stride, 2, 2),
                          padding=(conv1_t_size // 2, 3, 3),
                          bias=False)
        bn1 = nn.BatchNorm3d(self.in_planes)
        relu = nn.ReLU(inplace=True)

        if not no_max_pool:
            maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
            self.conv_in = nn.Sequential(conv1, bn1, relu, maxpool)
        else:
            self.conv_in = nn.Sequential(conv1, bn1, relu)

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
                                       stride=1, dilation=2)

        self.transpose_layer = TranspConvNet(in_channels=512, classes=n_classes)

        self.init_weights()

    def init_weights(self):
        # for reproducibility
        torch.manual_seed(777)
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
        if self.device() == "cuda":
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        """
        :param block: either Basic or BottleNeck block since they shae the same paprameters
        :param planes: out channels
        """
        downsample = None
        assert shortcut_type in ["A", "B"]

        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            # shortcut B
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
            layers.append(block(self.in_planes, planes, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # segmentation transp conv here
        y = self.transpose_layer(x)
        return y

    def test(self):
        input_tensor = torch.rand(1, self.input_channels, 32, 32, 32)
        ideal_target = torch.rand(1, self.n_classes, 32, 32, 32)
        out = self.forward(input_tensor)
        print(out.shape)


def generate_3d_res_net_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]
    res_net_dict = {10: [1, 1, 1, 1], 18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3],
                    152: [3, 8, 36, 3], 200: [3, 24, 36, 3]}

    in_planes = [64, 128, 256, 512]

    if model_depth == 10:
        model = ResNet(BasicBlock, res_net_dict[model_depth], in_planes, **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, res_net_dict[model_depth], in_planes, **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, res_net_dict[model_depth], in_planes, **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, res_net_dict[model_depth], in_planes, **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, res_net_dict[model_depth], in_planes, **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, res_net_dict[model_depth], in_planes, **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, res_net_dict[model_depth], in_planes, **kwargs)

    return model



model = generate_3d_res_net_model(10)
print(model.device())
model.test()

