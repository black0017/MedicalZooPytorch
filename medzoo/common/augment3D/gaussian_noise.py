import numpy as np
from .apply_augmentations import Augment

def random_noise(img_numpy, noise):
    """

    Args:
        img_numpy:
        mean: mean value of
        std:

    Returns:
        image with added random noise

    """


    return img_numpy + noise


class GaussianNoise(Augment):
    """

    """
    def __init__(self, modality_keys, apply_to_label=False, mean=0, std=0.001):
        super(GaussianNoise, self).__init__(modality_keys, apply_to_label)
        self.mean = mean
        self.std = std

    def __call__(self, data):
        noise = np.random.normal(self.mean, self.std, data[self.modality_keys[0]].shape)
        for key in self.modality_keys:
            if key != 'label':
                data[key] = random_noise(data[key], noise)

        return data
