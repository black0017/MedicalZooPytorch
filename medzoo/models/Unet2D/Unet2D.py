import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from medzoo.models.BaseModelClass import BaseModel


# 2D-Unet Model taken from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py

class _DoubleConv(nn.Module):
    """
    Building block with 2 3x3 convolutions with batch norm and relu
    """

    def __init__(self, in_ch, out_ch):
        super(_DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class _InConv(nn.Module):
    """
    The module to process the input medical volume
    """

    def __init__(self, in_ch, out_ch):
        super(_InConv, self).__init__()
        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x):
        self.conv(x)


class _Down(nn.Module):
    """
    Halves the spatial dim with max pooling
    Then 2x Conv are applied
    """

    def __init__(self, in_ch, out_ch):
        super(_Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            _DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.mpconv(x)


class _Up(nn.Module):
    """
    Upsampling block with bilinear upsampling
    Transpose convolutions is also supported by setting bilinear to True
    """
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(_Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class _OutConv(nn.Module):
    """

    """

    def __init__(self, in_ch, out_ch):
        super(_OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        """

        Args:
            x:

        Returns:

        """
        x = self.conv(x)
        return x


class Unet(BaseModel):
    """
    Based on the original paper: https://arxiv.org/abs/1505.04597


    """

    def __init__(self, in_channels, classes):
        super(Unet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = classes

        self.inc = _InConv(in_channels, 64)
        self.down1 = _Down(64, 128)
        self.down2 = _Down(128, 256)
        self.down3 = _Down(256, 512)
        self.down4 = _Down(512, 512)
        self.up1 = _Up(1024, 256)
        self.up2 = _Up(512, 128)
        self.up3 = _Up(256, 64)
        self.up4 = _Up(128, 64)
        self.outc = _OutConv(64, classes)

    def forward(self, x):
        """
        Args:
            x: 5D Tensor of shape [batch, channels, slices/depth, height, width]

        Returns: an identical shape 5D tensor with channels=number of classes
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

    def test(self, device='cpu'):
        device = torch.device(device)
        input_tensor = torch.rand(1, self.n_channels, 32, 32)
        ideal_out = torch.rand(1, self.n_classes, 32, 32)
        out = self.forward(input_tensor)
        assert ideal_out.shape == out.shape
        summary(self.to(device), (self.n_channels, 32, 32, 32), device=device)
        # import torchsummaryX
        # torchsummaryX.summary(self, input_tensor.to(device))
        print("Unet 2D test is complete")
