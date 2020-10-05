import random
from abc import ABC, abstractmethod

import numpy as np

functions = ['elastic_deform', 'random_crop', 'random_flip', 'random_rescale', 'random_rotate', 'random_shift']


class Augment(ABC):

    def __init__(self, modality_keys, apply_to_label):
        """

        Args:
            modality_keys (list): List of data
            apply_to_label ():
        """
        self.modality_keys = modality_keys
        self.apply_to_label = apply_to_label

    @abstractmethod
    def __call__(self, data):
        """

        """
        raise NotImplementedError


class RandomAugment(Augment):
    def __init__(self, modality_keys, apply_to_label):
        super(RandomAugment, self).__init__(modality_keys, apply_to_label)

    def __call__(self, input):
        raise NotImplementedError

    def set_random(self, **args):
        raise NotImplementedError


class Compose(object):
    def __init__(self, transforms=[]):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


class RandomChoice(object):
    """Choose a random tranform from list and apply

    Args:
        transforms: tranforms to apply
        p: probability

    """

    def __init__(self, transforms=[],
                 p=0.5):
        self.transforms = transforms
        self.p = p

    def __call__(self, img_tensors, label):
        augment = np.random.random(1) < self.p
        if not augment:
            return img_tensors, label
        t = random.choice(self.transforms)

        for i in range(len(img_tensors)):

            if i == (len(img_tensors) - 1):
                ### do only once the augmentation to the label
                img_tensors[i], label = t(img_tensors[i], label)
            else:
                img_tensors[i], _ = t(img_tensors[i], label)
        return img_tensors, label


class ComposeTransforms(object):
    """
    Composes several transforms together.
    """

    def __init__(self, transforms=[],
                 p=0.9):
        self.transforms = transforms
        self.p = p

    def __call__(self, img_tensors, label):
        augment = np.random.random(1) < self.p
        if not augment:
            return img_tensors, label

        for i in range(len(img_tensors)):

            for t in self.transforms:
                if i == (len(img_tensors) - 1):
                    ### do only once augmentation to the label
                    img_tensors[i], label = t(img_tensors[i], label)
                else:
                    img_tensors[i], _ = t(img_tensors[i], label)
        return img_tensors, label
