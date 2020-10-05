import numpy as np
import scipy.ndimage as ndimage
from .apply_augmentations import RandomAugment


def random_rotate3D(img_numpy, angle, axes):
    """ Returns a random rotated array in the same shape

    Args:
        img_numpy: 3D numpy array
        min_angle:  in degrees
        max_angle: in degrees

    Returns:
         3D rotated img
    """

    assert img_numpy.ndim == 3, "provide a 3d numpy array"

    return ndimage.rotate(img_numpy, angle, axes=axes)


class RandomRotation(RandomAugment):
    """

    """

    def __init__(self, modality_keys, apply_to_label=True, min_angle=-10, max_angle=10):
        super(RandomRotation, self).__init__(modality_keys, apply_to_label)
        assert min_angle < max_angle, "min should be less than max val"
        assert min_angle > -360 or max_angle < 360
        self.min_angle = min_angle
        self.max_angle = max_angle

    def set_random(self, **args):
        all_axes = [(1, 0), (1, 2), (0, 2)]

        angle = np.random.randint(low=self.min_angle, high=self.max_angle + 1)
        axes_random_id = np.random.randint(low=0, high=len(all_axes))
        axes = all_axes[axes_random_id]
        return angle, axes

    def __call__(self, data):
        """
        Args:
            img_numpy (numpy): Image to be rotated.
            label (numpy): Label segmentation map to be rotated

        Returns:
            img_numpy (numpy): rotated img.
            label (numpy): rotated Label segmentation map.
        """
        angle, axes = self.set_random()
        for key in self.modality_keys:
            data[key] = random_rotate3D(data[key], self.min_angle, self.max_angle)
        if self.apply_to_label:
            data['label'] = random_rotate3D(data['label'], self.min_angle, self.max_angle)

        return data
