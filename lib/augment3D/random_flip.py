import numpy as np


# TODO test

def random_flip(img_numpy, label=None):
    axes = [0, 1, 2]
    rand = np.random.randint(0, 3)
    img_numpy = flip_axis(img_numpy, axes[rand])
    img_numpy = np.squeeze(img_numpy)

    if label is None:
        return img_numpy
    else:
        y = flip_axis(label, axes[rand])
        y = np.squeeze(y)
    return img_numpy, y


def flip_axis(img_numpy, axis):
    img_numpy = np.asarray(img_numpy).swapaxes(axis, 0)
    img_numpy = img_numpy[::-1, ...]
    img_numpy = img_numpy.swapaxes(0, axis)
    return img_numpy
