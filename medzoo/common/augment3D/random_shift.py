import numpy as np
import scipy.ndimage as ndimage

from .apply_augmentations import RandomAugment


def transform_matrix_offset_center_3d(matrix, x, y, z):
    """

    Args:
        matrix:
        x:
        y:
        z:

    Returns:

    """
    offset_matrix = np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
    return ndimage.interpolation.affine_transform(matrix, offset_matrix)


def random_shift(img_numpy, max_percentage=0.2):
    """

    Args:
        img_numpy:
        max_percentage:

    Returns:

    """
    dim1, dim2, dim3 = img_numpy.shape
    m1, m2, m3 = int(dim1 * max_percentage / 2), int(dim1 * max_percentage / 2), int(dim1 * max_percentage / 2)
    d1 = np.random.randint(-m1, m1)
    d2 = np.random.randint(-m2, m2)
    d3 = np.random.randint(-m3, m3)
    return transform_matrix_offset_center_3d(img_numpy, d1, d2, d3)


def random_shift2(img_numpy, d1,d2,d3):
    """

    Args:
        img_numpy:
        max_percentage:

    Returns:

    """

    return transform_matrix_offset_center_3d(img_numpy, d1, d2, d3)


class RandomShift(RandomAugment):
    """

    """

    def __init__(self, modality_keys, apply_to_label=True, max_percentage=0.2):
        super(RandomShift, self).__init__(modality_keys, apply_to_label)
        self.max_percentage = max_percentage

    # @staticmethod
    # def get_params(img, output_size):

    def set_random(self,dim1):
        m1, m2, m3 = int(dim1 * self.max_percentage / 2), int(dim1 * self.max_percentage / 2), int(dim1 * self.max_percentage / 2)
        d1 = np.random.randint(-m1, m1)
        d2 = np.random.randint(-m2, m2)
        d3 = np.random.randint(-m3, m3)
        return
    def __call__(self, data):

        img_numpy = random_shift(img_numpy, self.max_percentage)
        if label.any() != None:
            label = random_shift(label, self.max_percentage)
        return img_numpy, label
