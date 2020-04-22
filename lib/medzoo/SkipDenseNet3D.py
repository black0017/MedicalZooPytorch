from collections import OrderedDict

import torch
import torch.nn as nn
from torchsummary import summary

from lib.medzoo.BaseModelClass import BaseModel

"""
Based on the implementation of https://github.com/tbuikr/3D-SkipDenseSeg
Paper here : https://arxiv.org/pdf/1709.03199.pdf
"""


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        if self.drop_rate > 0:
            self.drop_layer = nn.Dropout(p=self.drop_rate)

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = self.drop_layer(new_features)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))

        self.add_module('pool_norm', nn.BatchNorm3d(num_output_features))
        self.add_module('pool_relu', nn.ReLU(inplace=True))
        self.add_module('pool', nn.Conv3d(num_output_features, num_output_features, kernel_size=2, stride=2))


# TODO test model
class SkipDenseNet3D(BaseModel):
    """Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Based on the implementation of https://github.com/tbuikr/3D-SkipDenseSeg
    Paper here : https://arxiv.org/pdf/1709.03199.pdf

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        classes (int) - number of classification classes
    """

    def __init__(self, in_channels=2, classes=4, growth_rate=16, block_config=(4, 4, 4, 4), num_init_features=32, drop_rate=0.1,
                 bn_size=4):

        super(SkipDenseNet3D, self).__init__()
        self.num_classes = classes
        # First three convolutions
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(in_channels, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('conv1', nn.Conv3d(num_init_features, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm1', nn.BatchNorm3d(num_init_features)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv3d(num_init_features, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
        ]))
        self.features_bn = nn.Sequential(OrderedDict([
            ('norm2', nn.BatchNorm3d(num_init_features)),
            ('relu2', nn.ReLU(inplace=True)),
        ]))
        self.conv_pool_first = nn.Conv3d(num_init_features, num_init_features, kernel_size=2, stride=2, padding=0,
                                         bias=False)

        # Each denseblock
        num_features = num_init_features
        self.dense_blocks = nn.ModuleList([])
        self.transit_blocks = nn.ModuleList([])
        self.upsampling_blocks = nn.ModuleList([])

        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)

            self.dense_blocks.append(block)

            num_features = num_features + num_layers * growth_rate

            up_block = nn.ConvTranspose3d(num_features, classes, kernel_size=2 ** (i + 1) + 2,
                                          stride=2 ** (i + 1),
                                          padding=1, groups=classes, bias=False)

            self.upsampling_blocks.append(up_block)

            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.transit_blocks.append(trans)
                # self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        # self.features.add_module('norm5', nn.BatchNorm3d(num_features))

        # Linear layer
        # self.classifier = nn.Linear(num_features, num_classes)
        # self.bn4 = nn.BatchNorm3d(num_features)

        # ----------------------- classifier -----------------------
        self.bn_class = nn.BatchNorm3d(classes * 4 + num_init_features)
        self.conv_class = nn.Conv3d(classes * 4 + num_init_features, classes, kernel_size=1, padding=0)
        self.relu_last = nn.ReLU()
        # ----------------------------------------------------------

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                # nn.Conv3d.bias.data.fill_(-0.1)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        first_three_features = self.features(x)
        first_three_features_bn = self.features_bn(first_three_features)
        out = self.conv_pool_first(first_three_features_bn)

        out = self.dense_blocks[0](out)
        up_block1 = self.upsampling_blocks[0](out)
        out = self.transit_blocks[0](out)

        out = self.dense_blocks[1](out)
        up_block2 = self.upsampling_blocks[1](out)
        out = self.transit_blocks[1](out)

        out = self.dense_blocks[2](out)
        up_block3 = self.upsampling_blocks[2](out)
        out = self.transit_blocks[2](out)

        out = self.dense_blocks[3](out)
        up_block4 = self.upsampling_blocks[3](out)

        out = torch.cat([up_block1, up_block2, up_block3, up_block4, first_three_features], 1)

        # ----------------------- classifier -----------------------
        out = self.conv_class(self.relu_last(self.bn_class(out)))
        # ----------------------------------------------------------
        return out

    def test(self,device='cpu'):

        input_tensor = torch.rand(1, 2, 32, 32, 32)
        ideal_out = torch.rand(1, self.num_classes, 32, 32, 32)
        out = self.forward(input_tensor)
        assert ideal_out.shape == out.shape
        summary(self.to(torch.device(device)), (2, 32, 32, 32),device=device)
        import torchsummaryX
        torchsummaryX.summary(self, input_tensor.to(device))
        print("SkipDenseNet3D test is complete")

# model = SkipDenseNet3D(growth_rate=16, num_init_features=32, drop_rate=0.1,           num_classes=4)
# model.test()
