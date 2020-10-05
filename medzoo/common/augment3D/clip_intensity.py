import numpy as np

from .apply_augmentations import Augment


def _percentile_clip(img_numpy, min_val=0.1, max_val=99.8):
    """Intensity normalization based on percentile
    Clips the range based on the quarile values.

    Args:
    min_val: should be in the range [0,100]
    max_val: should be in the range [0,100]

    Returns:
        intensity normalized image
    """
    low = np.percentile(img_numpy, min_val)
    high = np.percentile(img_numpy, max_val)

    img_numpy[img_numpy < low] = low
    img_numpy[img_numpy > high] = high
    return img_numpy


class ClipIntensity(Augment):
    def __init__(self, modality_keys, apply_to_label=False, min_val=0.1, max_val=99.8):
        """

        Args:
            modality_keys ():
            apply_to_label ():
            min_val ():
            max_val ():
        """
        super(ClipIntensity, self).__init__(modality_keys, apply_to_label)

        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, data):
        for key in self.modality_keys:
            if key!='label':
                data[key] = _percentile_clip(data[key], self.min_val, self.max_val)

        return data
