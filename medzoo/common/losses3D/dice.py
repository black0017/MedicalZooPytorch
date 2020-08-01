from medzoo.common.losses3D.BaseClass import _AbstractDiceLoss
from medzoo.common.losses3D.basic import *


# Code was adapted and mofified from https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/losses.py


class DiceLoss(_AbstractDiceLoss):
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.
    """

    def __init__(self, classes=4, skip_index_after=None, weight=None, sigmoid_normalization=True):
        super().__init__(weight, sigmoid_normalization)
        self.classes = classes
        if skip_index_after is not None:
            self.skip_index_after = skip_index_after

    def dice(self, input, target, weight):
        """

        Args:
            input:
            target:
            weight:

        Returns:

        """
        return compute_per_channel_dice(input, target, weight=self.weight)
