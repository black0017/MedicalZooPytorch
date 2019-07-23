import torch.nn as nn
import torch
import torch.nn.functional as F



"""
Implementations based on the Vnet paper: https://arxiv.org/pdf/1804.02967.pdf

"""

class UNet3D(nn.Module):
    def __init__(self, in_channels, n_classes, base_n_filter=8):
        super(UNet3D, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter

        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=0.6)
        self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
        self.softmax = nn.Softmax(dim=1)

        # Level 1 context pathway
        self.conv3d_c1_1 = nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1,
                                     bias=False)
        self.conv3d_c1_2 = nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1,
                                     bias=False)
        self.lrelu_conv_c1 = self.lrelu_conv(self.base_n_filter, self.base_n_filter)
        self.inorm3d_c1 = nn.InstanceNorm3d(self.base_n_filter)

        # Level 2 context pathway
        self.conv3d_c2 = nn.Conv3d(self.base_n_filter, self.base_n_filter * 2, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(self.base_n_filter * 2, self.base_n_filter * 2)
        self.inorm3d_c2 = nn.InstanceNorm3d(self.base_n_filter * 2)

        # Level 3 context pathway
        self.conv3d_c3 = nn.Conv3d(self.base_n_filter * 2, self.base_n_filter * 4, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(self.base_n_filter * 4, self.base_n_filter * 4)
        self.inorm3d_c3 = nn.InstanceNorm3d(self.base_n_filter * 4)

        # Level 4 context pathway
        self.conv3d_c4 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 8, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(self.base_n_filter * 8, self.base_n_filter * 8)
        self.inorm3d_c4 = nn.InstanceNorm3d(self.base_n_filter * 8)

        # Level 5 context pathway, level 0 localization pathway
        self.conv3d_c5 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 16, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c5 = self.norm_lrelu_conv(self.base_n_filter * 16, self.base_n_filter * 16)
        self.norm_lrelu_upscale_conv_norm_lrelu_l0 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 16,
                                                                                             self.base_n_filter * 8)

        self.conv3d_l0 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter * 8)

        # Level 1 localization pathway
        self.conv_norm_lrelu_l1 = self.conv_norm_lrelu(self.base_n_filter * 16, self.base_n_filter * 16)
        self.conv3d_l1 = nn.Conv3d(self.base_n_filter * 16, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l1 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 8,
                                                                                             self.base_n_filter * 4)

        # Level 2 localization pathway
        self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(self.base_n_filter * 8, self.base_n_filter * 8)
        self.conv3d_l2 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 4, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l2 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 4,
                                                                                             self.base_n_filter * 2)

        # Level 3 localization pathway
        self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(self.base_n_filter * 4, self.base_n_filter * 4)
        self.conv3d_l3 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 2, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l3 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 2,
                                                                                             self.base_n_filter)

        # Level 4 localization pathway
        self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(self.base_n_filter * 2, self.base_n_filter * 2)
        self.conv3d_l4 = nn.Conv3d(self.base_n_filter * 2, self.n_classes, kernel_size=1, stride=1, padding=0,
                                   bias=False)

        self.ds2_1x1_conv3d = nn.Conv3d(self.base_n_filter * 8, self.n_classes, kernel_size=1, stride=1, padding=0,
                                        bias=False)
        self.ds3_1x1_conv3d = nn.Conv3d(self.base_n_filter * 4, self.n_classes, kernel_size=1, stride=1, padding=0,
                                        bias=False)

    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # should be feat_in*2 or feat_in
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def forward(self, x):
        #  Level 1 context pathway
        out = self.conv3d_c1_1(x)
        residual_1 = out
        out = self.lrelu(out)
        out = self.conv3d_c1_2(out)
        out = self.dropout3d(out)
        out = self.lrelu_conv_c1(out)
        # Element Wise Summation
        out += residual_1
        context_1 = self.lrelu(out)
        out = self.inorm3d_c1(out)
        out = self.lrelu(out)

        # Level 2 context pathway
        out = self.conv3d_c2(out)
        residual_2 = out
        out = self.norm_lrelu_conv_c2(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c2(out)
        out += residual_2
        out = self.inorm3d_c2(out)
        out = self.lrelu(out)
        context_2 = out

        # Level 3 context pathway
        out = self.conv3d_c3(out)
        residual_3 = out
        out = self.norm_lrelu_conv_c3(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c3(out)
        out += residual_3
        out = self.inorm3d_c3(out)
        out = self.lrelu(out)
        context_3 = out

        # Level 4 context pathway
        out = self.conv3d_c4(out)
        residual_4 = out
        out = self.norm_lrelu_conv_c4(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c4(out)
        out += residual_4
        out = self.inorm3d_c4(out)
        out = self.lrelu(out)
        context_4 = out

        # Level 5
        out = self.conv3d_c5(out)
        residual_5 = out
        out = self.norm_lrelu_conv_c5(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c5(out)
        out += residual_5
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l0(out)

        out = self.conv3d_l0(out)
        out = self.inorm3d_l0(out)
        out = self.lrelu(out)

        # Level 1 localization pathway
        out = torch.cat([out, context_4], dim=1)
        out = self.conv_norm_lrelu_l1(out)
        out = self.conv3d_l1(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l1(out)

        # Level 2 localization pathway
        #print(out.shape)
        #print(context_3.shape)
        out = torch.cat([out, context_3], dim=1)
        out = self.conv_norm_lrelu_l2(out)
        ds2 = out
        out = self.conv3d_l2(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l2(out)

        # Level 3 localization pathway
        out = torch.cat([out, context_2], dim=1)
        out = self.conv_norm_lrelu_l3(out)
        ds3 = out
        out = self.conv3d_l3(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l3(out)

        # Level 4 localization pathway
        out = torch.cat([out, context_1], dim=1)
        out = self.conv_norm_lrelu_l4(out)
        out_pred = self.conv3d_l4(out)

        ds2_1x1_conv = self.ds2_1x1_conv3d(ds2)
        ds1_ds2_sum_upscale = self.upsacle(ds2_1x1_conv)
        ds3_1x1_conv = self.ds3_1x1_conv3d(ds3)
        ds1_ds2_sum_upscale_ds3_sum = ds1_ds2_sum_upscale + ds3_1x1_conv
        ds1_ds2_sum_upscale_ds3_sum_upscale = self.upsacle(ds1_ds2_sum_upscale_ds3_sum)

        out = out_pred + ds1_ds2_sum_upscale_ds3_sum_upscale
        seg_layer = out
        out = out.permute(0, 2, 3, 4, 1).contiguous()
        out = out.view(-1, self.n_classes)
        out = self.softmax(out)
        return seg_layer, out






"""
Implementations based on the Vnet paper: https://arxiv.org/pdf/1804.02967.pdf

"""

def passthrough(x, **kwargs):
    return x


def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)


class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)

        self.bn1 = torch.nn.BatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, elu):
        super(InputTransition, self).__init__()
        num_features = 16
        self.conv1 = nn.Conv3d(1, num_features, kernel_size=5, padding=2)

        self.bn1 = torch.nn.BatchNorm3d(num_features)

        self.relu1 = ELUCons(elu, num_features)

    def forward(self, x):
        out = self.conv1(x)

        out = self.bn1(out)
        # split input in to 16 channels !!!!!
        ## TODO use expand!!!
        x16 = torch.cat((x, x, x, x, x, x, x, x,
                        x, x, x, x, x, x, x, x), 1)
        """
        print("x shape", x.shape)
        print("x16 shape", x16.shape)
        print("out shape", out.shape)
        """

        return self.relu1(torch.add(out, x16))


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2 * inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = torch.nn.BatchNorm3d(outChans)

        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)

        self.bn1 = torch.nn.BatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, classes, elu, nll):
        super(OutputTransition, self).__init__()
        self.classes = classes
        self.conv1 = nn.Conv3d(inChans, classes, kernel_size=5, padding=2)
        self.bn1 = torch.nn.BatchNorm3d(classes)

        self.conv2 = nn.Conv3d(classes, classes, kernel_size=1)
        self.relu1 = ELUCons(elu, classes)
        if nll:
            self.softmax = F.log_softmax
        else:
            #self.softmax = F.softmax
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # convolve 32 down to channels as the desired classes
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)

        # make channels the last axis
        #out = out.permute(0, 2, 3, 4, 1).contiguous()
        # flatten
        #out = out.view(out.numel() // self.classes, self.classes) # out.view(-1,self.classes)
        #out = self.softmax(out)
        # treat channel 0 as the predicted output
        return out


class VNet(nn.Module):
    def __init__(self,elu=True, nll=False):
        super(VNet, self).__init__()
        self.in_tr = InputTransition(elu=elu)
        self.down_tr32 = DownTransition(16, 1, elu)
        self.down_tr64 = DownTransition(32, 2, elu)
        self.down_tr128 = DownTransition(64, 3, elu, dropout=True)
        self.down_tr256 = DownTransition(128, 2, elu, dropout=True)
        self.up_tr256 = UpTransition(256, 256, 2, elu, dropout=True)
        self.up_tr128 = UpTransition(256, 128, 2, elu, dropout=True)
        self.up_tr64 = UpTransition(128, 64, 1, elu)
        self.up_tr32 = UpTransition(64, 32, 1, elu)
        self.out_tr = OutputTransition(32, 4, elu, nll)

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out


class VNetLight(nn.Module):
    def __init__(self, elu=True, nll=False):
        super(VNetLight, self).__init__()
        self.debug = False
        self.in_tr = InputTransition(elu)
        self.down_tr32 = DownTransition(16, 1, elu)
        self.down_tr64 = DownTransition(32, 2, elu)
        self.down_tr128 = DownTransition(64, 3, elu, dropout=True)
        self.up_tr128 = UpTransition(128, 128, 2, elu, dropout=True)
        self.up_tr64 = UpTransition(128, 64, 1, elu)
        self.up_tr32 = UpTransition(64, 32, 1, elu)
        self.out_tr = OutputTransition(32, 4, elu, nll)

    def forward(self, x):
        out16 = self.in_tr(x)
        if self.debug:
            print("InputTransition DONE")
            print("out16", out16.shape)

        out32 = self.down_tr32(out16)
        if self.debug:
            print("DownTransition DONE - 1")
            print("out32", out32.shape)

        out64 = self.down_tr64(out32)

        if self.debug:
            print("DownTransition DONE - 2")
            print("out64", out64.shape)

        out128 = self.down_tr128(out64)

        if self.debug:
            print("DownTransition DONE - 3")
            print("out128", out128.shape)

        out = self.up_tr128(out128, out64)

        if self.debug:
            print("up Transition DONE - 6")
            print("out", out.shape)

        out = self.up_tr64(out, out32)

        if self.debug:
            print("up Transition DONE - 7")
            print("out", out.shape)

        out = self.up_tr32(out, out16)

        if self.debug:
            print("up Transition DONE - 8")
            print("out", out.shape)

        out = self.out_tr(out)

        if self.debug:
            print("Final")
        return out


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


class SinglePathDenseNet(nn.Module):

    def __init__(self, input_channels, drop_rate=0, num_classes=4, return_logits=True, early_fusion=False):
        super(SinglePathDenseNet, self).__init__()
        self.return_logits = return_logits
        self.features = nn.Sequential()
        self.classes = num_classes

        if early_fusion:
            block = _HyperDenseBlockEarlyFusion(num_input_features=input_channels, drop_rate=drop_rate)
            if input_channels == 52:
                total_conv_channels = 477
            else:
                total_conv_channels = 503

        else:
            block = _HyperDenseBlock(num_input_features=input_channels, drop_rate=drop_rate)
            if input_channels==2:
                total_conv_channels = 452
            else:
                total_conv_channels = 451

        self.features.add_module('denseblock1', block)

        self.features.add_module('conv1x1_1', nn.Conv3d(total_conv_channels,
                                                        400, kernel_size=1, stride=1, padding=0,
                                                        bias=False))

        self.features.add_module('drop_1', nn.Dropout(p=0.2))

        self.features.add_module('conv1x1_2', nn.Conv3d(400,
                                                        200, kernel_size=1, stride=1, padding=0,
                                                        bias=False))

        self.features.add_module('drop_2', nn.Dropout(p=0.2))

        self.features.add_module('conv1x1_3', nn.Conv3d(200,
                                                        150, kernel_size=1, stride=1, padding=0,
                                                        bias=False))

        self.features.add_module('drop_3', nn.Dropout(p=0.2))

        self.classifier = nn.Sequential()
        self.classifier.add_module('classifier', nn.Conv3d(150,
                                                           self.classes, kernel_size=1, stride=1, padding=0,
                                                           bias=False))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        features = self.features(x)
        if self.return_logits:
            out = self.classifier(features)
            # make channels the last axis
            #out = out.permute(0, 2, 3, 4, 1).contiguous()
            # flatten
            #out = out.view(-1, self.classes)
            #print("reshape out shape", out.shape)
            #out = self.softmax(out)

            return out

        else:
            return features


class DualPathDenseNet(nn.Module):

    def __init__(self, input_channels, drop_rate=0, num_classes=4, fusion='concat'):
        """
        2-stream and 3-stream implementation with late fusion
        :param input_channels: 2 or 3 (dual or triple path based on paper specifications).
        Channels are the input modalities i.e T1,T2 etc..
        :param drop_rate:  dropout rate for dense layers
        :param num_classes: number of classes to segment
        :param fusion: 'concat or 'sum'
        """
        super(DualPathDenseNet, self).__init__()
        self.input_channels = input_channels
        self.fusion = fusion
        if self.fusion == "concat":
            in_classifier_channels = self.input_channels * 150
        else:
            in_classifier_channels = 150

        if self.input_channels == 2:
            self.stream_1 = SinglePathDenseNet(input_channels=1, drop_rate=drop_rate, num_classes=num_classes,
                                               return_logits=False, early_fusion=True)
            self.stream_2 = SinglePathDenseNet(input_channels=1, drop_rate=drop_rate, num_classes=num_classes,
                                               return_logits=False, early_fusion=True)
            self.classifier = nn.Sequential()

        if self.input_channels == 3:
            self.stream_1 = SinglePathDenseNet(input_channels=1, drop_rate=drop_rate, num_classes=num_classes,
                                               return_logits=False)
            self.stream_2 = SinglePathDenseNet(input_channels=1, drop_rate=drop_rate, num_classes=num_classes,
                                               return_logits=False)
            self.stream_3 = SinglePathDenseNet(input_channels=1, drop_rate=drop_rate, num_classes=num_classes,
                                               return_logits=False)

        self.classifier = nn.Sequential()
        self.classifier.add_module('classifier', nn.Conv3d(in_classifier_channels,
                                                           num_classes, kernel_size=1, stride=1, padding=0,
                                                           bias=False))

    def forward(self, multi_channel_medical_img):
        """
        :param multi_channel_medical_img: shape of [batch, input_channels, height, width, depth]
        :return: late fusion classification predictions
        """
        print("med img shape", multi_channel_medical_img.shape)
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


class DualSingleDensenet(nn.Module):
    """
    2-stream and 3-stream implementation with early fusion
    dual-single-densenet OR Disentangled modalities with early fusion in the paper
    """

    def __init__(self, input_channels, drop_rate=0, num_classes=4):
        """

        :param input_channels: 2 or 3 (dual or triple path based on paper specifications).
        Channels are the input modalities i.e T1,T2 etc..
        :param drop_rate:  dropout rate for dense layers
        :param num_classes: number of classes to segment
        :param fusion: 'concat or 'sum'
        """
        super(DualSingleDensenet, self).__init__()
        self.input_channels = input_channels
        in_classifier_channels = self.input_channels * 150

        if self.input_channels == 2:
            self.early_conv_1 = _HyperDenseLayer(num_input_features=1, num_output_channels=25, drop_rate=0)
            self.early_conv_2 = _HyperDenseLayer(num_input_features=1, num_output_channels=25, drop_rate=0)
            single_path_channels = 52
            self.stream_1 = SinglePathDenseNet(input_channels=single_path_channels, drop_rate=drop_rate,
                                               num_classes=num_classes, return_logits=True, early_fusion=True)
            self.classifier = nn.Sequential()

        if self.input_channels == 3:
            self.early_conv_1 = _HyperDenseLayer(num_input_features=1, num_output_channels=25, drop_rate=0)
            self.early_conv_2 = _HyperDenseLayer(num_input_features=1, num_output_channels=25, drop_rate=0)
            self.early_conv_3 = _HyperDenseLayer(num_input_features=1, num_output_channels=25, drop_rate=0)
            single_path_channels = 78
            self.stream_1 = SinglePathDenseNet(input_channels=single_path_channels, drop_rate=drop_rate,
                                               num_classes=num_classes, return_logits=True, early_fusion=True)

    def forward(self, multi_channel_medical_img):
        """
        :param multi_channel_medical_img: shape of [batch, input_channels, height, width, depth]
        :return: late fusion classification predictions
        """
        #print("med img shape", multi_channel_medical_img.shape)
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





class DiceLoss(nn.Module):
    """Computes Dice Loss, which just 1 - DiceCoefficient described above.
    Additionally allows per-class weights to be provided.
            # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify sigmoid_normalization=False.
    """

    def __init__(self, epsilon=1e-5, sigmoid_normalization=True,
                 skip_last_target=False, idx_to_ignore_after=None):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
        if idx_to_ignore_after is not None:
            self.idx_to_ignore_after = idx_to_ignore_after
        else:
            self.idx_to_ignore_after = None

        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)
        # if True skip the last channel in the target
        self.skip_last_target = skip_last_target

    def flatten(self,tensor):
        """Flattens a given tensor such that the channel axis is first.
        The shapes are transformed as follows:
           (N, C, D, H, W) -> (C, N * D * H * W)
        """
        C = tensor.size(1)
        # new axis order
        axis_order = (1, 0) + tuple(range(2, tensor.dim()))
        # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
        transposed = tensor.permute(axis_order)
        # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
        return transposed.view(C, -1)

    def expand_as_one_hot(self, input, C, ignore_index=None):
        """
        Converts NxDxHxW label image to NxCxDxHxW, where each label gets converted to its corresponding one-hot vector
        :param input: 4D input image (NxDxHxW)
        :param C: number of channels/labels
        :param ignore_index: ignore index to be kept during the expansion
        :return: 5D output image (NxCxDxHxW)
        """
        assert input.dim() == 4

        shape = input.size()
        shape = list(shape)
        shape.insert(1, C)
        shape = tuple(shape)

        # expand the input tensor to Nx1xDxHxW
        src = input.unsqueeze(0)

        if ignore_index is not None:
            # create ignore_index mask for the result
            expanded_src = src.expand(shape)
            mask = expanded_src == ignore_index
            # clone the src tensor and zero out ignore_index in the input
            src = src.clone()
            src[src == ignore_index] = 0
            # scatter to get the one-hot tensor
            result = torch.zeros(shape).to(input.device).scatter_(1, src, 1)
            # bring back the ignore_index in the result
            result[mask] = ignore_index
            return result
        else:
            # scatter to get the one-hot tensor
            return torch.zeros(shape).to(input.device).scatter_(1, src, 1)

    def compute_per_channel_dice(self, input, target):
        epsilon=1e-5
        # assumes that input is a normalized probability

        # input and target shapes must match
        target = self.expand_as_one_hot(target.long(),4
                                        )
        if self.idx_to_ignore_after is not None:
            target = target.clone()[:,:self.idx_to_ignore_after,...]
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        input = self.flatten(input)
        target = self.flatten(target)

        target = target.float()
        # Compute per channel Dice Coefficient
        intersect = (input * target).sum(-1)

        denominator = (input + target).sum(-1)
        return 2. * intersect / denominator.clamp(min=epsilon)

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)
        per_channel_dice = self.compute_per_channel_dice(input, target)
        temp = per_channel_dice.clone().cpu()
        DSC = temp.detach().numpy()
        # Average the Dice score across all channels/classes
        return torch.mean(1. - per_channel_dice), DSC

