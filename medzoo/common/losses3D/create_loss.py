import torch.nn as nn
from torch.nn import MSELoss, SmoothL1Loss, L1Loss

from .BCE_dice import BCEDiceLoss
from .dice import DiceLoss
from .generalized_dice import GeneralizedDiceLoss
from .pixel_wise_cross_entropy import PixelWiseCrossEntropyLoss
from .tags_angular_loss import TagsAngularLoss
from .weight_cross_entropy import WeightedCrossEntropyLoss
from .weight_smooth_l1 import WeightedSmoothL1Loss

# Code was adapted and mofified from https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/losses.py



def create_loss(name, weight=None, ignore_index=None, pos_weight=None):
    """

    Args:
        name: the name of the loss function to be used
        weight:
        ignore_index:
        pos_weight:

    Returns: the supported loss function

    """
    SUPPORTED_LOSSES = ['BCEWithLogitsLoss', 'BCEDiceLoss', 'CrossEntropyLoss', 'WeightedCrossEntropyLoss',
                        'PixelWiseCrossEntropyLoss', 'GeneralizedDiceLoss', 'DiceLoss', 'TagsAngularLoss', 'MSELoss',
                        'SmoothL1Loss', 'L1Loss', 'WeightedSmoothL1Loss']

    if name == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif name == 'BCEDiceLoss':
        return BCEDiceLoss(alpha=1, beta=1)
    elif name == 'CrossEntropyLoss':
        if ignore_index is None:
            ignore_index = -100  # use the default 'ignore_index' as defined in the CrossEntropyLoss
        return nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
    elif name == 'WeightedCrossEntropyLoss':
        if ignore_index is None:
            ignore_index = -100  # use the default 'ignore_index' as defined in the CrossEntropyLoss
        return WeightedCrossEntropyLoss(ignore_index=ignore_index)
    elif name == 'PixelWiseCrossEntropyLoss':
        return PixelWiseCrossEntropyLoss(class_weights=weight, ignore_index=ignore_index)
    elif name == 'GeneralizedDiceLoss':

        return GeneralizedDiceLoss(sigmoid_normalization=False)
    elif name == 'DiceLoss':
        return DiceLoss(weight=weight, sigmoid_normalization=False)
    elif name == 'TagsAngularLoss':
        return TagsAngularLoss()
    elif name == 'MSELoss':
        return MSELoss()
    elif name == 'SmoothL1Loss':
        return SmoothL1Loss()
    elif name == 'L1Loss':
        return L1Loss()
    elif name == 'WeightedSmoothL1Loss':
        return WeightedSmoothL1Loss()
    else:
        raise RuntimeError(f"Unsupported loss function: '{name}'. Supported losses: {SUPPORTED_LOSSES}")
