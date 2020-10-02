import random
import numpy as np


functions = ['elastic_deform', 'random_crop', 'random_flip', 'random_rescale', 'random_rotate', 'random_shift']

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
