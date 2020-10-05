import numpy as np


def random_crop_to_labels(img_numpy, label):
    """
    Random center crop near the label area
    """
    target_indexs = np.where(label > 0)
    [img_d, img_h, img_w] = img_numpy.shape
    [max_D, max_H, max_W] = np.max(np.array(target_indexs), axis=1)
    [min_D, min_H, min_W] = np.min(np.array(target_indexs), axis=1)
    [target_depth, target_height, target_width] = np.array([max_D, max_H, max_W]) - np.array([min_D, min_H, min_W])
    Z_min = int((min_D - target_depth * 1.0 / 2) * np.random.sample())
    Y_min = int((min_H - target_height * 1.0 / 2) * np.random.sample())
    X_min = int((min_W - target_width * 1.0 / 2) * np.random.sample())

    Z_max = int(img_d - ((img_d - (max_D + target_depth * 1.0 / 2)) * np.random.sample()))
    Y_max = int(img_h - ((img_h - (max_H + target_height * 1.0 / 2)) * np.random.sample()))
    X_max = int(img_w - ((img_w - (max_W + target_width * 1.0 / 2)) * np.random.sample()))

    Z_min = int(np.max([0, Z_min]))
    Y_min = int(np.max([0, Y_min]))
    X_min = int(np.max([0, X_min]))

    Z_max = int(np.min([img_d, Z_max]))
    Y_max = int(np.min([img_h, Y_max]))
    X_max = int(np.min([img_w, X_max]))

    return img_numpy[Z_min: Z_max, Y_min: Y_max, X_min: X_max], label


class RandomCropToLabels(object):

    def __call__(self, img_numpy, label):
        return random_crop_to_labels(img_numpy, label)
