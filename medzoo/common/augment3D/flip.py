import numpy as np

from .apply_augmentations import RandomAugment, Augment


# TODO test

def random_flip(img_numpy, axis_for_flip=0):
    """

    Args:
        img_numpy:
        label:
        axis_for_flip:

    Returns:

    """


    img_numpy = flip_axis(img_numpy, axis_for_flip)
    img_numpy = np.squeeze(img_numpy)
    return img_numpy


def flip_axis(img_numpy, axis):
    """

    Args:
        img_numpy:
        axis:

    Returns:

    """
    img_numpy = np.asarray(img_numpy).swapaxes(axis, 0)
    img_numpy = img_numpy[::-1, ...]
    img_numpy = img_numpy.swapaxes(0, axis)
    return img_numpy


class FlipAxis(Augment):
    def __init__(self, modality_keys, apply_to_label=True, select_flip_axis=0):
        super(FlipAxis, self).__init__(modality_keys, apply_to_label)
        self.flip_axis = select_flip_axis

    def __call__(self, data):

        for key in self.modality_keys:
            if key != 'label':
                data[key] = random_flip(data[key], self.flip_axis)
        if self.apply_to_label:
            data['label'] = random_flip(data['label'], self.flip_axis)

        return data


class RandomFlip(RandomAugment):

    def __init__(self, modality_keys, apply_to_label=True):
        """

        Args:
            modality_keys ():
            apply_to_label ():
        """
        super(RandomFlip, self).__init__(modality_keys, apply_to_label)


    def set_random(self):

        return np.random.randint(0, 3)

    def __call__(self, data):
        """
        Apply random axis flip to data
        Args:
            data (dict):  Dictionary of numpy arrays

        Returns:

        """
        axis_for_flip = self.set_random()
        for key in self.modality_keys:
            if key!='label':
                data[key] = random_flip(data[key], axis_for_flip)
        if self.apply_to_label:
            data['label'] = random_flip(data['label'], axis_for_flip)

        return data
