import torch
import torch.nn as nn
from torchsummary import summary
from lib.medzoo.BaseModelClass import BaseModel

"""
Code was borrowed and modified from this repo: https://github.com/josedolz/HyperDenseNet_pytorch 
"""


def conv(nin, nout, kernel_size=3, stride=1, padding=1, bias=False, layer=nn.Conv2d,
         BN=False, ws=False, activ=nn.LeakyReLU(0.2), gainWS=2):
    convlayer = layer(nin, nout, kernel_size, stride=stride, padding=padding, bias=bias)
    layers = []
    # if ws:
    #     layers.append(WScaleLayer(convlayer, gain=gainWS))
    if BN:
        layers.append(nn.BatchNorm2d(nout))
    if activ is not None:
        if activ == nn.PReLU:
            # to avoid sharing the same parameter, activ must be set to nn.PReLU (without '()')
            layers.append(activ(num_parameters=1))
        else:
            # if activ == nn.PReLU(), the parameter will be shared for the whole network !
            layers.append(activ)
    layers.insert(ws, convlayer)
    return nn.Sequential(*layers)


class ResidualConv(nn.Module):
    def __init__(self, nin, nout, bias=False, BN=False, ws=False, activ=nn.LeakyReLU(0.2)):
        super(ResidualConv, self).__init__()

        convs = [conv(nin, nout, bias=bias, BN=BN, ws=ws, activ=activ),
                 conv(nout, nout, bias=bias, BN=BN, ws=ws, activ=None)]
        self.convs = nn.Sequential(*convs)

        res = []
        if nin != nout:
            res.append(conv(nin, nout, kernel_size=1, padding=0, bias=False, BN=BN, ws=ws, activ=None))
        self.res = nn.Sequential(*res)

        activation = []
        if activ is not None:
            if activ == nn.PReLU:
                # to avoid sharing the same parameter, activ must be set to nn.PReLU (without '()')
                activation.append(activ(num_parameters=1))
            else:
                # if activ == nn.PReLU(), the parameter will be shared for the whole network !
                activation.append(activ)
        self.activation = nn.Sequential(*activation)

    def forward(self, input):
        out = self.convs(input)
        return self.activation(out + self.res(input))


def upSampleConv_Res(nin, nout, upscale=2, bias=False, BN=False, ws=False, activ=nn.LeakyReLU(0.2)):
    return nn.Sequential(
        nn.Upsample(scale_factor=upscale),
        ResidualConv(nin, nout, bias=bias, BN=BN, ws=ws, activ=activ)
    )


def conv_block(in_dim, out_dim, act_fn, kernel_size=3, stride=1, padding=1, dilation=1):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model


def conv_block_1(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=1),
        nn.BatchNorm2d(out_dim),
        nn.PReLU(),
    )
    return model


def conv_block_Asym(in_dim, out_dim, kernelSize):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=[kernelSize, 1], padding=tuple([2, 0])),
        nn.Conv2d(out_dim, out_dim, kernel_size=[1, kernelSize], padding=tuple([0, 2])),
        nn.BatchNorm2d(out_dim),
        nn.PReLU(),
    )
    return model


def conv_block_Asym_Inception(in_dim, out_dim, kernel_size, padding, dilation=1):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=[kernel_size, 1], padding=tuple([padding * dilation, 0]),
                  dilation=(dilation, 1)),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=[1, kernel_size], padding=tuple([0, padding * dilation]),
                  dilation=(dilation, 1)),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
    )
    return model


def conv_block_Asym_Inception_WithIncreasedFeatMaps(in_dim, mid_dim, out_dim, kernel_size, padding, dilation=1):
    model = nn.Sequential(
        nn.Conv2d(in_dim, mid_dim, kernel_size=[kernel_size, 1], padding=tuple([padding * dilation, 0]),
                  dilation=(dilation, 1)),
        nn.BatchNorm2d(mid_dim),
        nn.ReLU(),
        nn.Conv2d(mid_dim, out_dim, kernel_size=[1, kernel_size], padding=tuple([0, padding * dilation]),
                  dilation=(dilation, 1)),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
    )
    return model


def conv_block_Asym_ERFNet(in_dim, out_dim, kernelSize, padding, drop, dilation):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=[kernelSize, 1], padding=tuple([padding, 0]), bias=True),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=[1, kernelSize], padding=tuple([0, padding]), bias=True),
        nn.BatchNorm2d(out_dim, eps=1e-03),
        nn.ReLU(),
        nn.Conv2d(in_dim, out_dim, kernel_size=[kernelSize, 1], padding=tuple([padding * dilation, 0]), bias=True,
                  dilation=(dilation, 1)),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=[1, kernelSize], padding=tuple([0, padding * dilation]), bias=True,
                  dilation=(1, dilation)),
        nn.BatchNorm2d(out_dim, eps=1e-03),
        nn.Dropout2d(drop),
    )
    return model


def conv_block_3_3(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_dim),
        nn.PReLU(),
    )
    return model


# TODO: Change order of block: BN + Activation + Conv
def conv_decod_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model


def dilation_conv_block(in_dim, out_dim, act_fn, stride_val, dil_val):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride_val, padding=1, dilation=dil_val),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model


def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool


def avrgpool05():
    pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    return pool


def avrgpool025():
    pool = nn.AvgPool2d(kernel_size=2, stride=4, padding=0)
    return pool


def avrgpool0125():
    pool = nn.AvgPool2d(kernel_size=2, stride=8, padding=0)
    return pool


def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool


def maxpool_1_4():
    pool = nn.MaxPool2d(kernel_size=2, stride=4, padding=0)
    return pool


def maxpool_1_8():
    pool = nn.MaxPool2d(kernel_size=2, stride=8, padding=0)
    return pool


def maxpool_1_16():
    pool = nn.MaxPool2d(kernel_size=2, stride=16, padding=0)
    return pool


def maxpool_1_32():
    pool = nn.MaxPool2d(kernel_size=2, stride=32, padding=0)


def conv_block_3(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        conv_block(in_dim, out_dim, act_fn),
        conv_block(out_dim, out_dim, act_fn),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
    )
    return model


def classificationNet(D_in):
    H = 400
    D_out = 1
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, int(H / 4)),
        torch.nn.ReLU(),
        torch.nn.Linear(int(H / 4), D_out)
    )

    return model


# from layers import *

def croppCenter(tensorToCrop, finalShape):
    org_shape = tensorToCrop.shape
    diff = org_shape[2] - finalShape[2]
    croppBorders = int(diff / 2)
    return tensorToCrop[:,
           :,
           croppBorders:org_shape[2] - croppBorders,
           croppBorders:org_shape[3] - croppBorders,
           croppBorders:org_shape[4] - croppBorders]


def convBlock(nin, nout, kernel_size=3, batchNorm=False, layer=nn.Conv3d, bias=True, dropout_rate=0.0, dilation=1):
    if batchNorm == False:
        return nn.Sequential(
            nn.PReLU(),
            nn.Dropout(p=dropout_rate),
            layer(nin, nout, kernel_size=kernel_size, bias=bias, dilation=dilation)
        )
    else:
        return nn.Sequential(
            nn.BatchNorm3d(nin),
            nn.PReLU(),
            nn.Dropout(p=dropout_rate),
            layer(nin, nout, kernel_size=kernel_size, bias=bias, dilation=dilation)
        )


def convBatch(nin, nout, kernel_size=3, stride=1, padding=1, bias=False, layer=nn.Conv2d, dilation=1):
    return nn.Sequential(
        layer(nin, nout, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, dilation=dilation),
        nn.BatchNorm2d(nout),
        # nn.LeakyReLU(0.2)
        nn.PReLU()
    )


class HyperDenseNet_2Mod(BaseModel):
    def __init__(self, in_channels=2, classes=4):
        super(HyperDenseNet_2Mod, self).__init__()
        self.num_classes = classes
        assert in_channels == 2, "input channels must be two for this architecture"

        # Path-Top
        self.conv1_Top = convBlock(1, 25)
        self.conv2_Top = convBlock(50, 25, batchNorm=True)
        self.conv3_Top = convBlock(100, 25, batchNorm=True)
        self.conv4_Top = convBlock(150, 50, batchNorm=True)
        self.conv5_Top = convBlock(250, 50, batchNorm=True)
        self.conv6_Top = convBlock(350, 50, batchNorm=True)
        self.conv7_Top = convBlock(450, 75, batchNorm=True)
        self.conv8_Top = convBlock(600, 75, batchNorm=True)
        self.conv9_Top = convBlock(750, 75, batchNorm=True)

        # Path-Bottom
        self.conv1_Bottom = convBlock(1, 25)
        self.conv2_Bottom = convBlock(50, 25, batchNorm=True)
        self.conv3_Bottom = convBlock(100, 25, batchNorm=True)
        self.conv4_Bottom = convBlock(150, 50, batchNorm=True)
        self.conv5_Bottom = convBlock(250, 50, batchNorm=True)
        self.conv6_Bottom = convBlock(350, 50, batchNorm=True)
        self.conv7_Bottom = convBlock(450, 75, batchNorm=True)
        self.conv8_Bottom = convBlock(600, 75, batchNorm=True)
        self.conv9_Bottom = convBlock(750, 75, batchNorm=True)

        self.fully_1 = nn.Conv3d(1800, 400, kernel_size=1)
        self.fully_2 = nn.Conv3d(400, 200, kernel_size=1)
        self.fully_3 = nn.Conv3d(200, 150, kernel_size=1)
        self.final = nn.Conv3d(150, classes, kernel_size=1)

    def forward(self, input):
        # ----- First layer ------ #
        # get 2 of the channels as 5D tensors
        # pdb.set_trace()
        print("input shape ", input.shape)
        y1t = self.conv1_Top(input[:, 0:1, :, :, :])
        y1b = self.conv1_Bottom(input[:, 1:2, :, :, :])

        # ----- Second layer ------ #
        # concatenate
        y2t_i = torch.cat((y1t, y1b), dim=1)
        y2b_i = torch.cat((y1b, y1t), dim=1)

        y2t_o = self.conv2_Top(y2t_i)
        y2b_o = self.conv2_Bottom(y2b_i)

        # ----- Third layer ------ #
        y2t_i_cropped = croppCenter(y2t_i, y2t_o.shape)
        y2b_i_cropped = croppCenter(y2b_i, y2t_o.shape)

        # concatenate
        y3t_i = torch.cat((y2t_i_cropped, y2t_o, y2b_o), dim=1)
        y3b_i = torch.cat((y2b_i_cropped, y2b_o, y2t_o), dim=1)

        y3t_o = self.conv3_Top(y3t_i)
        y3b_o = self.conv3_Bottom(y3b_i)

        # ------ Fourth layer ----- #
        y3t_i_cropped = croppCenter(y3t_i, y3t_o.shape)
        y3b_i_cropped = croppCenter(y3b_i, y3t_o.shape)

        # concatenate
        y4t_i = torch.cat((y3t_i_cropped, y3t_o, y3b_o), dim=1)
        y4b_i = torch.cat((y3b_i_cropped, y3b_o, y3t_o), dim=1)

        y4t_o = self.conv4_Top(y4t_i)
        y4b_o = self.conv4_Bottom(y4b_i)

        # ------ Fifth layer ----- #
        y4t_i_cropped = croppCenter(y4t_i, y4t_o.shape)
        y4b_i_cropped = croppCenter(y4b_i, y4t_o.shape)

        # concatenate
        y5t_i = torch.cat((y4t_i_cropped, y4t_o, y4b_o), dim=1)
        y5b_i = torch.cat((y4b_i_cropped, y4b_o, y4t_o), dim=1)

        y5t_o = self.conv5_Top(y5t_i)
        y5b_o = self.conv5_Bottom(y5b_i)

        # ------ Sixth layer ----- #
        y5t_i_cropped = croppCenter(y5t_i, y5t_o.shape)
        y5b_i_cropped = croppCenter(y5b_i, y5t_o.shape)

        # concatenate
        y6t_i = torch.cat((y5t_i_cropped, y5t_o, y5b_o), dim=1)
        y6b_i = torch.cat((y5b_i_cropped, y5b_o, y5t_o), dim=1)

        y6t_o = self.conv6_Top(y6t_i)
        y6b_o = self.conv6_Bottom(y6b_i)

        # ------ Seventh layer ----- #
        y6t_i_cropped = croppCenter(y6t_i, y6t_o.shape)
        y6b_i_cropped = croppCenter(y6b_i, y6t_o.shape)

        # concatenate
        y7t_i = torch.cat((y6t_i_cropped, y6t_o, y6b_o), dim=1)
        y7b_i = torch.cat((y6b_i_cropped, y6b_o, y6t_o), dim=1)

        y7t_o = self.conv7_Top(y7t_i)
        y7b_o = self.conv7_Bottom(y7b_i)

        # ------ Eight layer ----- #
        y7t_i_cropped = croppCenter(y7t_i, y7t_o.shape)
        y7b_i_cropped = croppCenter(y7b_i, y7t_o.shape)

        # concatenate
        y8t_i = torch.cat((y7t_i_cropped, y7t_o, y7b_o), dim=1)
        y8b_i = torch.cat((y7b_i_cropped, y7b_o, y7t_o), dim=1)

        y8t_o = self.conv8_Top(y8t_i)
        y8b_o = self.conv8_Bottom(y8b_i)

        # ------ Ninth layer ----- #
        y8t_i_cropped = croppCenter(y8t_i, y8t_o.shape)
        y8b_i_cropped = croppCenter(y8b_i, y8t_o.shape)

        # concatenate
        y9t_i = torch.cat((y8t_i_cropped, y8t_o, y8b_o), dim=1)
        y9b_i = torch.cat((y8b_i_cropped, y8b_o, y8t_o), dim=1)

        y9t_o = self.conv9_Top(y9t_i)
        y9b_o = self.conv9_Bottom(y9b_i)

        ##### Fully connected layers
        y9t_i_cropped = croppCenter(y9t_i, y9t_o.shape)
        y9b_i_cropped = croppCenter(y9b_i, y9t_o.shape)

        outputPath_top = torch.cat((y9t_i_cropped, y9t_o, y9b_o), dim=1)
        outputPath_bottom = torch.cat((y9b_i_cropped, y9b_o, y9t_o), dim=1)

        inputFully = torch.cat((outputPath_top, outputPath_bottom), dim=1)

        y = self.fully_1(inputFully)
        y = self.fully_2(y)
        y = self.fully_3(y)

        return self.final(y)

    def test(self, device='cpu'):
        input_tensor = torch.rand(1, 2, 22, 22, 22)
        ideal_out = torch.rand(1, self.num_classes, 22, 22, 22)
        out = self.forward(input_tensor)
        # assert ideal_out.shape == out.shape
        # summary(self.to(torch.device(device)), (2, 22, 22, 22),device=device)
        # torchsummaryX.summary(self,input_tensor.to(device))
        print("HyperDenseNet test is complete", out.shape)


class HyperDenseNet(BaseModel):
    def __init__(self, in_channels=3, classes=4):
        super(HyperDenseNet, self).__init__()
        assert in_channels == 3, "HyperDensenet supports 3 in_channels. For 2 in_channels use HyperDenseNet_2Mod "
        self.num_classes = classes

        # Path-Top
        self.conv1_Top = convBlock(1, 25)
        self.conv2_Top = convBlock(75, 25, batchNorm=True)
        self.conv3_Top = convBlock(150, 25, batchNorm=True)
        self.conv4_Top = convBlock(225, 50, batchNorm=True)
        self.conv5_Top = convBlock(375, 50, batchNorm=True)
        self.conv6_Top = convBlock(525, 50, batchNorm=True)
        self.conv7_Top = convBlock(675, 75, batchNorm=True)
        self.conv8_Top = convBlock(900, 75, batchNorm=True)
        self.conv9_Top = convBlock(1125, 75, batchNorm=True)

        # Path-Middle
        self.conv1_Middle = convBlock(1, 25)
        self.conv2_Middle = convBlock(75, 25, batchNorm=True)
        self.conv3_Middle = convBlock(150, 25, batchNorm=True)
        self.conv4_Middle = convBlock(225, 50, batchNorm=True)
        self.conv5_Middle = convBlock(375, 50, batchNorm=True)
        self.conv6_Middle = convBlock(525, 50, batchNorm=True)
        self.conv7_Middle = convBlock(675, 75, batchNorm=True)
        self.conv8_Middle = convBlock(900, 75, batchNorm=True)
        self.conv9_Middle = convBlock(1125, 75, batchNorm=True)

        # Path-Bottom
        self.conv1_Bottom = convBlock(1, 25)
        self.conv2_Bottom = convBlock(75, 25, batchNorm=True)
        self.conv3_Bottom = convBlock(150, 25, batchNorm=True)
        self.conv4_Bottom = convBlock(225, 50, batchNorm=True)
        self.conv5_Bottom = convBlock(375, 50, batchNorm=True)
        self.conv6_Bottom = convBlock(525, 50, batchNorm=True)
        self.conv7_Bottom = convBlock(675, 75, batchNorm=True)
        self.conv8_Bottom = convBlock(900, 75, batchNorm=True)
        self.conv9_Bottom = convBlock(1125, 75, batchNorm=True)

        self.fully_1 = nn.Conv3d(4050, 400, kernel_size=1)
        self.fully_2 = nn.Conv3d(400, 200, kernel_size=1)
        self.fully_3 = nn.Conv3d(200, 150, kernel_size=1)
        self.final = nn.Conv3d(150, classes, kernel_size=1)

    def forward(self, input):
        # ----- First layer ------ #
        # get the 3 channels as 5D tensors
        y1t = self.conv1_Top(input[:, 0:1, :, :, :])
        y1m = self.conv1_Middle(input[:, 1:2, :, :, :])
        y1b = self.conv1_Bottom(input[:, 2:3, :, :, :])

        # ----- Second layer ------ #
        # concatenate
        y2t_i = torch.cat((y1t, y1m, y1b), dim=1)
        y2m_i = torch.cat((y1m, y1t, y1b), dim=1)
        y2b_i = torch.cat((y1b, y1t, y1m), dim=1)

        y2t_o = self.conv2_Top(y2t_i)
        y2m_o = self.conv2_Middle(y2m_i)
        y2b_o = self.conv2_Bottom(y2b_i)

        # ----- Third layer ------ #
        y2t_i_cropped = croppCenter(y2t_i, y2t_o.shape)
        y2m_i_cropped = croppCenter(y2m_i, y2t_o.shape)
        y2b_i_cropped = croppCenter(y2b_i, y2t_o.shape)

        # concatenate
        y3t_i = torch.cat((y2t_i_cropped, y2t_o, y2m_o, y2b_o), dim=1)
        y3m_i = torch.cat((y2m_i_cropped, y2m_o, y2t_o, y2b_o), dim=1)
        y3b_i = torch.cat((y2b_i_cropped, y2b_o, y2t_o, y2m_o), dim=1)

        y3t_o = self.conv3_Top(y3t_i)
        y3m_o = self.conv3_Middle(y3m_i)
        y3b_o = self.conv3_Bottom(y3b_i)

        # ------ Fourth layer ----- #
        y3t_i_cropped = croppCenter(y3t_i, y3t_o.shape)
        y3m_i_cropped = croppCenter(y3m_i, y3t_o.shape)
        y3b_i_cropped = croppCenter(y3b_i, y3t_o.shape)

        # concatenate
        y4t_i = torch.cat((y3t_i_cropped, y3t_o, y3m_o, y3b_o), dim=1)
        y4m_i = torch.cat((y3m_i_cropped, y3m_o, y3t_o, y3b_o), dim=1)
        y4b_i = torch.cat((y3b_i_cropped, y3b_o, y3t_o, y3m_o), dim=1)

        y4t_o = self.conv4_Top(y4t_i)
        y4m_o = self.conv4_Middle(y4m_i)
        y4b_o = self.conv4_Bottom(y4b_i)

        # ------ Fifth layer ----- #
        y4t_i_cropped = croppCenter(y4t_i, y4t_o.shape)
        y4m_i_cropped = croppCenter(y4m_i, y4t_o.shape)
        y4b_i_cropped = croppCenter(y4b_i, y4t_o.shape)

        # concatenate
        y5t_i = torch.cat((y4t_i_cropped, y4t_o, y4m_o, y4b_o), dim=1)
        y5m_i = torch.cat((y4m_i_cropped, y4m_o, y4t_o, y4b_o), dim=1)
        y5b_i = torch.cat((y4b_i_cropped, y4b_o, y4t_o, y4m_o), dim=1)

        y5t_o = self.conv5_Top(y5t_i)
        y5m_o = self.conv5_Middle(y5m_i)
        y5b_o = self.conv5_Bottom(y5b_i)

        # ------ Sixth layer ----- #
        y5t_i_cropped = croppCenter(y5t_i, y5t_o.shape)
        y5m_i_cropped = croppCenter(y5m_i, y5t_o.shape)
        y5b_i_cropped = croppCenter(y5b_i, y5t_o.shape)

        # concatenate
        y6t_i = torch.cat((y5t_i_cropped, y5t_o, y5m_o, y5b_o), dim=1)
        y6m_i = torch.cat((y5m_i_cropped, y5m_o, y5t_o, y5b_o), dim=1)
        y6b_i = torch.cat((y5b_i_cropped, y5b_o, y5t_o, y5m_o), dim=1)

        y6t_o = self.conv6_Top(y6t_i)
        y6m_o = self.conv6_Middle(y6m_i)
        y6b_o = self.conv6_Bottom(y6b_i)

        # ------ Seventh layer ----- #
        y6t_i_cropped = croppCenter(y6t_i, y6t_o.shape)
        y6m_i_cropped = croppCenter(y6m_i, y6t_o.shape)
        y6b_i_cropped = croppCenter(y6b_i, y6t_o.shape)

        # concatenate
        y7t_i = torch.cat((y6t_i_cropped, y6t_o, y6m_o, y6b_o), dim=1)
        y7m_i = torch.cat((y6m_i_cropped, y6m_o, y6t_o, y6b_o), dim=1)
        y7b_i = torch.cat((y6b_i_cropped, y6b_o, y6t_o, y6m_o), dim=1)

        y7t_o = self.conv7_Top(y7t_i)
        y7m_o = self.conv7_Middle(y7m_i)
        y7b_o = self.conv7_Bottom(y7b_i)

        # ------ Eight layer ----- #
        y7t_i_cropped = croppCenter(y7t_i, y7t_o.shape)
        y7m_i_cropped = croppCenter(y7m_i, y7t_o.shape)
        y7b_i_cropped = croppCenter(y7b_i, y7t_o.shape)

        # concatenate
        y8t_i = torch.cat((y7t_i_cropped, y7t_o, y7m_o, y7b_o), dim=1)
        y8m_i = torch.cat((y7m_i_cropped, y7m_o, y7t_o, y7b_o), dim=1)
        y8b_i = torch.cat((y7b_i_cropped, y7b_o, y7t_o, y7m_o), dim=1)

        y8t_o = self.conv8_Top(y8t_i)
        y8m_o = self.conv8_Middle(y8m_i)
        y8b_o = self.conv8_Bottom(y8b_i)

        # ------ Ninth layer ----- #
        y8t_i_cropped = croppCenter(y8t_i, y8t_o.shape)
        y8m_i_cropped = croppCenter(y8m_i, y8t_o.shape)
        y8b_i_cropped = croppCenter(y8b_i, y8t_o.shape)

        # concatenate
        y9t_i = torch.cat((y8t_i_cropped, y8t_o, y8m_o, y8b_o), dim=1)
        y9m_i = torch.cat((y8m_i_cropped, y8m_o, y8t_o, y8b_o), dim=1)
        y9b_i = torch.cat((y8b_i_cropped, y8b_o, y8t_o, y8m_o), dim=1)

        y9t_o = self.conv9_Top(y9t_i)
        y9m_o = self.conv9_Middle(y9m_i)
        y9b_o = self.conv9_Bottom(y9b_i)

        ##### Fully connected layers
        y9t_i_cropped = croppCenter(y9t_i, y9t_o.shape)
        y9m_i_cropped = croppCenter(y9m_i, y9t_o.shape)
        y9b_i_cropped = croppCenter(y9b_i, y9t_o.shape)

        outputPath_top = torch.cat((y9t_i_cropped, y9t_o, y9m_o, y9b_o), dim=1)
        outputPath_middle = torch.cat((y9m_i_cropped, y9m_o, y9t_o, y9b_o), dim=1)
        outputPath_bottom = torch.cat((y9b_i_cropped, y9b_o, y9t_o, y9m_o), dim=1)

        inputFully = torch.cat((outputPath_top, outputPath_middle, outputPath_bottom), dim=1)

        y = self.fully_1(inputFully)
        y = self.fully_2(y)
        y = self.fully_3(y)

        return self.final(y)

    def test(self, device='cpu'):
        device = torch.device(device)
        input_tensor = torch.rand(1, 3, 20, 20, 20)
        ideal_out = torch.rand(1, self.num_classes, 20, 20, 20)
        out = self.forward(input_tensor)
        # assert ideal_out.shape == out.shape
        summary(self, (3, 16, 16, 16))
        # torchsummaryX.summary(self, input_tensor.to(device))
        print("HyperDenseNet test is complete!!!", out.shape)

# m = HyperDenseNet(1,4)
# m.test()
