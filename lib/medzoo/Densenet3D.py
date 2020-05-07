import torch.nn as nn
import torch
import torch.nn.functional as F
from torchsummary import summary
from lib.medzoo.BaseModelClass import BaseModel

"""
Implementations based on the HyperDenseNet paper: https://arxiv.org/pdf/1804.02967.pdf
"""


class _HyperDenseLayer(nn.Sequential):
    def __init__(self, num_input_features, num_output_channels, drop_rate):
        super(_HyperDenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features,
                                           num_output_channels, kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_HyperDenseLayer, self).forward(x)

        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)

        return torch.cat([x, new_features], 1)


class _HyperDenseBlock(nn.Sequential):
    """
    Constructs a series of dense-layers based on in and out kernels list
    """

    def __init__(self, num_input_features, drop_rate):
        super(_HyperDenseBlock, self).__init__()
        out_kernels = [1, 25, 25, 25, 50, 50, 50, 75, 75, 75]
        self.number_of_conv_layers = 9

        in_kernels = [num_input_features]
        for j in range(1, len(out_kernels)):
            temp = in_kernels[-1]
            in_kernels.append(temp + out_kernels[j])

        print("out:", out_kernels)
        print("in:", in_kernels)

        for i in range(self.number_of_conv_layers):
            layer = _HyperDenseLayer(in_kernels[i], out_kernels[i + 1], drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _HyperDenseBlockEarlyFusion(nn.Sequential):
    def __init__(self, num_input_features, drop_rate):
        super(_HyperDenseBlockEarlyFusion, self).__init__()
        out_kernels = [1, 25, 25, 50, 50, 50, 75, 75, 75]
        self.number_of_conv_layers = 8

        in_kernels = [num_input_features]
        for j in range(1, len(out_kernels)):
            temp = in_kernels[-1]
            in_kernels.append(temp + out_kernels[j])

        print("out:", out_kernels)
        print("in:", in_kernels)

        for i in range(self.number_of_conv_layers):
            layer = _HyperDenseLayer(in_kernels[i], out_kernels[i + 1], drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class SinglePathDenseNet(BaseModel):
    def __init__(self, in_channels, classes=4, drop_rate=0.1, return_logits=True, early_fusion=False):
        super(SinglePathDenseNet, self).__init__()
        self.return_logits = return_logits
        self.features = nn.Sequential()
        self.num_classes = classes
        self.input_channels = in_channels

        if early_fusion:
            block = _HyperDenseBlockEarlyFusion(num_input_features=in_channels, drop_rate=drop_rate)
            if in_channels == 52:
                total_conv_channels = 477
            else:
                if in_channels == 3:
                    total_conv_channels = 426
                else:
                    total_conv_channels = 503

        else:
            block = _HyperDenseBlock(num_input_features=in_channels, drop_rate=drop_rate)
            if in_channels == 2:
                total_conv_channels = 452
            else:
                total_conv_channels = 451

        self.features.add_module('denseblock1', block)

        self.features.add_module('conv1x1_1', nn.Conv3d(total_conv_channels,
                                                        400, kernel_size=1, stride=1, padding=0,
                                                        bias=False))

        self.features.add_module('drop_1', nn.Dropout(p=0.5))

        self.features.add_module('conv1x1_2', nn.Conv3d(400,
                                                        200, kernel_size=1, stride=1, padding=0,
                                                        bias=False))

        self.features.add_module('drop_2', nn.Dropout(p=0.5))

        self.features.add_module('conv1x1_3', nn.Conv3d(200,
                                                        150, kernel_size=1, stride=1, padding=0,
                                                        bias=False))

        self.features.add_module('drop_3', nn.Dropout(p=0.5))

        self.classifier = nn.Sequential()
        self.classifier.add_module('classifier', nn.Conv3d(150,
                                                           self.num_classes, kernel_size=1, stride=1, padding=0,
                                                           bias=False))

    def forward(self, x):
        features = self.features(x)
        if self.return_logits:
            out = self.classifier(features)
            return out

        else:
            return features

    def test(self,device='cpu'):

        input_tensor = torch.rand(1, self.input_channels, 12, 12, 12)
        ideal_out = torch.rand(1, self.num_classes, 12, 12, 12)
        out = self.forward(input_tensor)
        assert ideal_out.shape == out.shape
        summary(self.to(torch.device(device)), (self.input_channels, 12, 12, 12),device=device)
        # import torchsummaryX
        # torchsummaryX.summary(self, input_tensor.to(device))
        print("DenseNet3D-1 test is complete")


class DualPathDenseNet(BaseModel):
    def __init__(self, in_channels, classes=4, drop_rate=0, fusion='concat'):
        """
        2-stream and 3-stream implementation with late fusion
        :param in_channels: 2 or 3 (dual or triple path based on paper specifications).
        Channels are the input modalities i.e T1,T2 etc..
        :param drop_rate:  dropout rate for dense layers
        :param classes: number of classes to segment
        :param fusion: 'concat or 'sum'
        """
        super(DualPathDenseNet, self).__init__()
        self.input_channels = in_channels
        self.num_classes = classes

        self.fusion = fusion
        if self.fusion == "concat":
            in_classifier_channels = self.input_channels * 150
        else:
            in_classifier_channels = 150

        if self.input_channels == 2:
            # here!!!!
            self.stream_1 = SinglePathDenseNet(in_channels=1, drop_rate=drop_rate, classes=classes,
                                               return_logits=False, early_fusion=True)
            self.stream_2 = SinglePathDenseNet(in_channels=1, drop_rate=drop_rate, classes=classes,
                                               return_logits=False, early_fusion=True)

        if self.input_channels == 3:
            self.stream_1 = SinglePathDenseNet(in_channels=1, drop_rate=drop_rate, classes=classes,
                                               return_logits=False)
            self.stream_2 = SinglePathDenseNet(in_channels=1, drop_rate=drop_rate, classes=classes,
                                               return_logits=False)
            self.stream_3 = SinglePathDenseNet(in_channels=1, drop_rate=drop_rate, classes=classes,
                                               return_logits=False)

        self.classifier = nn.Sequential()
        self.classifier.add_module('classifier', nn.Conv3d(in_classifier_channels,
                                                           classes, kernel_size=1, stride=1, padding=0,
                                                           bias=False))

    def forward(self, multi_channel_medical_img):
        """
        :param multi_channel_medical_img: shape of [batch, input_channels, height, width, depth]
        :return: late fusion classification predictions
        """
        channels = multi_channel_medical_img.shape[1]
        if channels != self.input_channels:
            print("Network channels does not match input channels, check your model/input!")
            return None
        else:
            if self.input_channels == 2:
                in_stream_1 = multi_channel_medical_img[:, 0, ...].unsqueeze(dim=1)
                in_stream_2 = multi_channel_medical_img[:, 1, ...].unsqueeze(dim=1)
                output_features_t1 = self.stream_1(in_stream_1)
                output_features_t2 = self.stream_2(in_stream_2)

                if self.fusion == 'concat':
                    concat_features = torch.cat((output_features_t1, output_features_t2), dim=1)
                    return self.classifier(concat_features)
                else:
                    features = output_features_t1 + output_features_t2
                    return self.classifier(features)
            elif self.input_channels == 3:
                in_stream_1 = multi_channel_medical_img[:, 0, ...].unsqueeze(dim=1)
                in_stream_2 = multi_channel_medical_img[:, 1, ...].unsqueeze(dim=1)
                in_stream_3 = multi_channel_medical_img[:, 2, ...].unsqueeze(dim=1)
                output_features_t1 = self.stream_1(in_stream_1)
                output_features_t2 = self.stream_2(in_stream_2)
                output_features_t3 = self.stream_3(in_stream_3)
                if self.fusion == 'concat':
                    concat_features = torch.cat((output_features_t1, output_features_t2, output_features_t3), dim=1)
                    return self.classifier(concat_features)
                else:
                    features = output_features_t1 + output_features_t2 + output_features_t3
                    return self.classifier(features)

    def test(self,device='cpu'):
        input_tensor = torch.rand(1, self.input_channels, 12, 12, 12)
        ideal_out = torch.rand(1, self.num_classes, 12, 12, 12)
        out = self.forward(input_tensor)
        assert ideal_out.shape == out.shape
        summary(self.to(torch.device(device)), (self.input_channels, 12, 12, 12),device=device)
        import torchsummaryX
        torchsummaryX.summary(self, input_tensor.to(device))
        print("DenseNet3D-2 test is complete!!!!\n\n\n\n\n")


class DualSingleDenseNet(BaseModel):
    """
    2-stream and 3-stream implementation with early fusion
    dual-single-densenet OR Disentangled modalities with early fusion in the paper
    """

    def __init__(self, in_channels, classes=4, drop_rate=0.5,):
        """

        :param input_channels: 2 or 3 (dual or triple path based on paper specifications).
        Channels are the input modalities i.e T1,T2 etc..
        :param drop_rate:  dropout rate for dense layers
        :param classes: number of classes to segment
        :param fusion: 'concat or 'sum'
        """
        super(DualSingleDenseNet, self).__init__()
        self.input_channels = in_channels
        self.num_classes = classes

        if self.input_channels == 2:
            self.early_conv_1 = _HyperDenseLayer(num_input_features=1, num_output_channels=25, drop_rate=drop_rate)
            self.early_conv_2 = _HyperDenseLayer(num_input_features=1, num_output_channels=25, drop_rate=drop_rate)
            single_path_channels = 52
            self.stream_1 = SinglePathDenseNet(in_channels=single_path_channels, drop_rate=drop_rate,
                                               classes=classes, return_logits=True, early_fusion=True)
            self.classifier = nn.Sequential()

        if self.input_channels == 3:
            self.early_conv_1 = _HyperDenseLayer(num_input_features=1, num_output_channels=25, drop_rate=0)
            self.early_conv_2 = _HyperDenseLayer(num_input_features=1, num_output_channels=25, drop_rate=0)
            self.early_conv_3 = _HyperDenseLayer(num_input_features=1, num_output_channels=25, drop_rate=0)
            single_path_channels = 78
            self.stream_1 = SinglePathDenseNet(in_channels=single_path_channels, drop_rate=drop_rate,
                                               classes=classes, return_logits=True, early_fusion=True)

    def forward(self, multi_channel_medical_img):
        """
        :param multi_channel_medical_img: shape of [batch, input_channels, height, width, depth]
        :return: late fusion classification predictions
        """
        channels = multi_channel_medical_img.shape[1]
        if channels != self.input_channels:
            print("Network channels does not match input channels, check your model/input!")
            return None
        else:
            if self.input_channels == 2:
                in_1 = multi_channel_medical_img[:, 0, ...].unsqueeze(dim=1)
                in_2 = multi_channel_medical_img[:, 1, ...].unsqueeze(dim=1)
                y1 = self.early_conv_1(in_1)
                y2 = self.early_conv_1(in_2)
                print(y1.shape)
                print(y2.shape)
                in_stream = torch.cat((y1, y2), dim=1)
                logits = self.stream_1(in_stream)
                return logits

            elif self.input_channels == 3:
                in_1 = multi_channel_medical_img[:, 0, ...].unsqueeze(dim=1)
                in_2 = multi_channel_medical_img[:, 1, ...].unsqueeze(dim=1)
                in_3 = multi_channel_medical_img[:, 2, ...].unsqueeze(dim=1)
                y1 = self.early_conv_1(in_1)
                y2 = self.early_conv_2(in_2)
                y3 = self.early_conv_3(in_3)
                in_stream = torch.cat((y1, y2, y3), dim=1)
                logits = self.stream_1(in_stream)
                return logits

    def test(self,device='cpu'):

        input_tensor = torch.rand(1, self.input_channels, 12, 12, 12)
        ideal_out = torch.rand(1, self.num_classes, 12, 12, 12)
        out = self.forward(input_tensor)
        assert ideal_out.shape == out.shape
        summary(self.to(torch.device(device)), (self.input_channels, 12, 12, 12),device=device)

        # import torchsummaryX
        # torchsummaryX.summary(self, input_tensor.to(device))
        print("DenseNet3D-3 test is complete\n\n")
