import torch
import torch.nn as nn
from torch.nn import MSELoss, SmoothL1Loss, L1Loss

from .BCE_dice import BCEDiceLoss
from .ContrastiveLoss import ContrastiveLoss
from .Dice2D import DiceLoss2D
from .dice import DiceLoss
from .generalized_dice import GeneralizedDiceLoss
from .pixel_wise_cross_entropy import PixelWiseCrossEntropyLoss
from .tags_angular_loss import TagsAngularLoss
from .weight_cross_entropy import WeightedCrossEntropyLoss
from .weight_smooth_l1 import WeightedSmoothL1Loss
from .create_loss import create_loss


