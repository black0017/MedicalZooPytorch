# __init__.py
import torch.optim as optim

from .Densenet3D import DualPathDenseNet, DualSingleDensenet, SinglePathDenseNet
from .Unet3D import UNet3D
from .Vnet import VNet, VNetLight
from .Dice import DiceLoss
from .Dice2D import DiceLoss2D
from .Unet2D import Unet
from .COVIDNet import CovidNet

def create_model(args):
    model_name = args.model
    optimizer_name = args.opt
    lr = args.lr
    in_channels = args.inChannels
    num_classes = args.classes
    weight_decay = 0.0000000005
    print("Building Model . . . . . . . ." + model_name)

    if model_name == 'VNET2':
        model = VNetLight(in_channels=in_channels, elu=False, nll=False, num_classes=num_classes)
    elif model_name == 'VNET':
        model = VNet(in_channels=in_channels, elu=False, nll=False, num_classes=num_classes)
    elif model_name == 'UNET3D':
        model = UNet3D(in_channels=in_channels, n_classes=num_classes, base_n_filter=8)
    elif model_name == 'DENSENET1':
        model = SinglePathDenseNet(input_channels=in_channels, num_classes=num_classes)
    elif model_name == 'DENSENET2':
        model = DualPathDenseNet(input_channels=in_channels, num_classes=num_classes)
    elif model_name == 'DENSENET3':
        model = DualSingleDensenet(input_channels=in_channels, drop_rate=0.1, num_classes=num_classes)
    elif model_name == "UNET2D":
        model = Unet(in_channels, num_classes)
    elif model_name == "COVIDNET":
        model = CovidNet(num_classes)


    print('Number of params: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    if optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

    return model, optimizer
