import numpy as np


def random_noise(img_numpy, mean=0, std=0.001):
    noise = np.random.normal(mean, std, img_numpy.shape)

    return img_numpy + noise


class GaussianNoise(object):
    def __init__(self, mean=0, std=0.001):
        self.mean = mean
        self.std = std

    def __call__(self, img_numpy, label=None):
        """
        Args:
            img_numpy (numpy): Image to be flipped.
            label (numpy): Label segmentation map to be flipped

        Returns:
            img_numpy (numpy):  flipped img.
            label (numpy): flipped Label segmentation.
        """

        return random_noise(img_numpy, self.mean, self.std), label
