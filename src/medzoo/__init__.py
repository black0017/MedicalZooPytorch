# __init__.py
import torch.optim as optim

from .Densenet3D import DualPathDenseNet, DualSingleDensenet, SinglePathDenseNet
from .Unet3D import UNet3D
from .Vnet import VNet, VNetLight
from .HyperDensenet import HyperDenseNet,HyperDenseNet_2Mod
from .Dice import DiceLoss
from .ResNet3D_VAE import *


def create_model(model_name, optimizer_name, lr, in_channels):
    weight_decay = 0.0000000005
    print("Building Model . . . . . . . ." + model_name)

    if model_name == 'VNET2':
        model = VNetLight(elu=False, nll=False)
    elif model_name == 'VNET':
        model = VNet(in_channels=in_channels, elu=False, nll=False)
    elif model_name == 'UNET3D':
        model = UNet3D(in_channels=in_channels, n_classes=4)
    elif model_name == 'DENSENET1':
        model = SinglePathDenseNet(input_channels=in_channels, num_classes=4)
    elif model_name == 'DENSENET2':
        model = DualPathDenseNet(input_channels=in_channels, num_classes=4)
    elif model_name == 'DENSENET3':
        model = DualSingleDensenet(input_channels=in_channels, drop_rate=0.1, num_classes=4)
    elif model_name == 'HYPER1':
        model = HyperDenseNet_2Mod(nClasses=4)
    elif model_name == 'HYPER':
        model = HyperDenseNet(nClasses=4)

    print('Number of params: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    if optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

    return model, optimizer
