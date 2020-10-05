import numpy as np

from .apply_augmentations import Augment


def _scale_intensity(img, min_intensity=0.0, max_intensity=1.0):
    """

    Args:
        img (numpy array): input array
        min_intensity (float): minimum value of the new scaled array
        max_intensity (float): maximum value of the new scaled array

    Returns:

    """
    min_img_value = np.min(img)
    max_img_value = np.max(img)

    if min_img_value == max_img_value:
        return img * min_img_value

    norm = (img - min_img_value) / (max_img_value - min_img_value)
    return (norm * (max_intensity - min_intensity)) + min_intensity


class ScaleIntensity(Augment):
    def __init__(self, modality_keys, apply_to_label=False, min_intensity=0.0, max_intensity=1.0):
        """
        Rescale array values to a new range between min_intensity and max_intensity
        Args:
            modality_keys ():
            apply_to_label (bool): Default: False
            min_intensity (float):
            max_intensity (float):
        """
        super(ScaleIntensity, self).__init__(modality_keys, apply_to_label)
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity

    def __call__(self, data):
        for key in self.modality_keys:
            data[key] = _scale_intensity(data[key], self.min_intensity, self.max_intensity)
        return data
