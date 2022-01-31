import torch.nn as nn

import torch.nn.functional as F
from torchvision import models


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class PEPX(nn.Module):
    def __init__(self, n_input, n_out):
        super(PEPX, self).__init__()

        '''
        • First-stage Projection: 1×1 convolutions for projecting input features to a lower dimension,

        • Expansion: 1×1 convolutions for expanding features
            to a higher dimension that is different than that of the
            input features,


        • Depth-wise Representation: efficient 3×3 depthwise convolutions for learning spatial characteristics to
            minimize computational complexity while preserving
            representational capacity,

        • Second-stage Projection: 1×1 convolutions for projecting features back to a lower dimension, and

        • Extension: 1×1 convolutions that finally extend channel dimensionality to a higher dimension to produce
             the final features.

        '''

        self.network = nn.Sequential(nn.Conv2d(in_channels=n_input, out_channels=n_input // 2, kernel_size=1),
                                     nn.Conv2d(in_channels=n_input // 2, out_channels=int(3 * n_input / 4),
                                               kernel_size=1),
                                     nn.Conv2d(in_channels=int(3 * n_input / 4), out_channels=int(3 * n_input / 4),
                                               kernel_size=3, groups=int(3 * n_input / 4), padding=1),
                                     nn.Conv2d(in_channels=int(3 * n_input / 4), out_channels=n_input // 2,
                                               kernel_size=1),
                                     nn.Conv2d(in_channels=n_input // 2, out_channels=n_out, kernel_size=1))

    def forward(self, x):
        return self.network(x)


class CovidNet(nn.Module):
    def __init__(self, model='large', n_classes=3):
        super(CovidNet, self).__init__()
        filters = {
            'pepx1_1': [64, 256],
            'pepx1_2': [256, 256],
            'pepx1_3': [256, 256],
            'pepx2_1': [256, 512],
            'pepx2_2': [512, 512],
            'pepx2_3': [512, 512],
            'pepx2_4': [512, 512],
            'pepx3_1': [512, 1024],
            'pepx3_2': [1024, 1024],
            'pepx3_3': [1024, 1024],
            'pepx3_4': [1024, 1024],
            'pepx3_5': [1024, 1024],
            'pepx3_6': [1024, 1024],
            'pepx4_1': [1024, 2048],
            'pepx4_2': [2048, 2048],
            'pepx4_3': [2048, 2048],
        }

        self.add_module('conv1', nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3))
        for key in filters:

            if ('pool' in key):
                self.add_module(key, nn.MaxPool2d(filters[key][0], filters[key][1]))
            else:
                self.add_module(key, PEPX(filters[key][0], filters[key][1]))

        if (model == 'large'):

            self.add_module('conv1_1x1', nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1))
            self.add_module('conv2_1x1', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1))
            self.add_module('conv3_1x1', nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1))
            self.add_module('conv4_1x1', nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1))

            self.__forward__ = self.forward_large_net
        else:
            self.__forward__ = self.forward_small_net
        self.add_module('flatten', Flatten())
        self.add_module('fc1', nn.Linear(7 * 7 * 2048, 1024))

        self.add_module('fc2', nn.Linear(1024, 256))
        self.add_module('classifier', nn.Linear(256, n_classes))

    def forward(self, x):
        return self.__forward__(x)

    def forward_large_net(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        out_conv1_1x1 = self.conv1_1x1(x)

        pepx11 = self.pepx1_1(x)
        pepx12 = self.pepx1_2(pepx11 + out_conv1_1x1)
        pepx13 = self.pepx1_3(pepx12 + pepx11 + out_conv1_1x1)

        out_conv2_1x1 = F.max_pool2d(self.conv2_1x1(pepx12 + pepx11 + pepx13 + out_conv1_1x1), 2)

        pepx21 = self.pepx2_1(
            F.max_pool2d(pepx13, 2) + F.max_pool2d(pepx11, 2) + F.max_pool2d(pepx12, 2) + F.max_pool2d(out_conv1_1x1,
                                                                                                       2))
        pepx22 = self.pepx2_2(pepx21 + out_conv2_1x1)
        pepx23 = self.pepx2_3(pepx22 + pepx21 + out_conv2_1x1)
        pepx24 = self.pepx2_4(pepx23 + pepx21 + pepx22 + out_conv2_1x1)

        out_conv3_1x1 = F.max_pool2d(self.conv3_1x1(pepx22 + pepx21 + pepx23 + pepx24 + out_conv2_1x1), 2)

        pepx31 = self.pepx3_1(
            F.max_pool2d(pepx24, 2) + F.max_pool2d(pepx21, 2) + F.max_pool2d(pepx22, 2) + F.max_pool2d(pepx23,
                                                                                                       2) + F.max_pool2d(
                out_conv2_1x1, 2))
        pepx32 = self.pepx3_2(pepx31 + out_conv3_1x1)
        pepx33 = self.pepx3_3(pepx31 + pepx32 + out_conv3_1x1)
        pepx34 = self.pepx3_4(pepx31 + pepx32 + pepx33 + out_conv3_1x1)
        pepx35 = self.pepx3_5(pepx31 + pepx32 + pepx33 + pepx34 + out_conv3_1x1)
        pepx36 = self.pepx3_6(pepx31 + pepx32 + pepx33 + pepx34 + pepx35 + out_conv3_1x1)

        out_conv4_1x1 = F.max_pool2d(
            self.conv4_1x1(pepx31 + pepx32 + pepx33 + pepx34 + pepx35 + pepx36 + out_conv3_1x1), 2)

        pepx41 = self.pepx4_1(
            F.max_pool2d(pepx31, 2) + F.max_pool2d(pepx32, 2) + F.max_pool2d(pepx32, 2) + F.max_pool2d(pepx34,
                                                                                                       2) + F.max_pool2d(
                pepx35, 2) + F.max_pool2d(pepx36, 2) + F.max_pool2d(out_conv3_1x1, 2))
        pepx42 = self.pepx4_2(pepx41 + out_conv4_1x1)
        pepx43 = self.pepx4_3(pepx41 + pepx42 + out_conv4_1x1)
        flattened = self.flatten(pepx41 + pepx42 + pepx43 + out_conv4_1x1)

        fc1out = F.relu(self.fc1(flattened))
        fc2out = F.relu(self.fc2(fc1out))
        logits = self.classifier(fc2out)
        return logits

    def forward_small_net(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)

        pepx11 = self.pepx1_1(x)
        pepx12 = self.pepx1_2(pepx11)
        pepx13 = self.pepx1_3(pepx12 + pepx11)

        pepx21 = self.pepx2_1(F.max_pool2d(pepx13, 2) + F.max_pool2d(pepx11, 2) + F.max_pool2d(pepx12, 2))
        pepx22 = self.pepx2_2(pepx21)
        pepx23 = self.pepx2_3(pepx22 + pepx21)
        pepx24 = self.pepx2_4(pepx23 + pepx21 + pepx22)

        pepx31 = self.pepx3_1(
            F.max_pool2d(pepx24, 2) + F.max_pool2d(pepx21, 2) + F.max_pool2d(pepx22, 2) + F.max_pool2d(pepx23, 2))
        pepx32 = self.pepx3_2(pepx31)
        pepx33 = self.pepx3_3(pepx31 + pepx32)
        pepx34 = self.pepx3_4(pepx31 + pepx32 + pepx33)
        pepx35 = self.pepx3_5(pepx31 + pepx32 + pepx33 + pepx34)
        pepx36 = self.pepx3_6(pepx31 + pepx32 + pepx33 + pepx34 + pepx35)

        pepx41 = self.pepx4_1(
            F.max_pool2d(pepx31, 2) + F.max_pool2d(pepx32, 2) + F.max_pool2d(pepx32, 2) + F.max_pool2d(pepx34,
                                                                                                       2) + F.max_pool2d(
                pepx35, 2) + F.max_pool2d(pepx36, 2))
        pepx42 = self.pepx4_2(pepx41)
        pepx43 = self.pepx4_3(pepx41 + pepx42)
        flattened = self.flatten(pepx41 + pepx42 + pepx43)

        fc1out = F.relu(self.fc1(flattened))
        fc2out = F.relu(self.fc2(fc1out))
        logits = self.classifier(fc2out)
        return logits


class CNN(nn.Module):
    def __init__(self, classes, model='resnet18'):
        super(CNN, self).__init__()
        if (model == 'resnet18'):
            self.cnn = models.resnet18(pretrained=True)
            self.cnn.fc = nn.Linear(512, classes)
        elif (model == 'resnext50_32x4d'):

            self.cnn = models.resnext50_32x4d(pretrained=True)
            self.cnn.classifier = nn.Linear(1280, classes)
        elif (model == 'mobilenet_v2'):

            self.cnn = models.mobilenet_v2(pretrained=True)
            self.cnn.classifier = nn.Linear(1280, classes)

    def forward(self, x):
        return self.cnn(x)
