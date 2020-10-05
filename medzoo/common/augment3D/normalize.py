import numpy as np
from .apply_augmentations import Augment

def _normalize_intensity(img_numpy, mean=None, std=None):
    """
    Accepts an image tensor and normalizes it by mean and std
    Args:
        img_numpy ():
        mean ():
        std ():

    Returns:

    """
    if mean is not None and std is not None:
        return (img_numpy - mean) / std
    else:
        return (img_numpy - np.mean(img_numpy)) / np.std(img_numpy)



class Normalize(Augment):
    """
    Converts the input image to a tensor
    """

    def __init__(
            self,modality_keys,apply_to_label=False, nonzero_only=False, channel_wise=False,mean=None, std=None):
        """

        Args:
            nonzero_only (bool): normalize only by non zero values
            channel_wise (): normalize only channelwise
        """
        super(Normalize, self).__init__(modality_keys, apply_to_label)

        self.nonzero_only = nonzero_only
        self.channel_wise = channel_wise
        self.mean = mean
        self.std = std

    def __call__(self, data):
        """
        Apply the transform to `img` and make it contiguous.

        Args:
            img (numpy array):
        """
        for key in self.modality_keys:
            if key!='label':

                if self.nonzero_only:
                    non_zero_img = (data[key] != 0)
                    data[key] = _normalize_intensity(data[key][non_zero_img],self.mean,self.std)
                else:

                    data[key] = _normalize_intensity(data[key],self.mean,self.std)
        return data