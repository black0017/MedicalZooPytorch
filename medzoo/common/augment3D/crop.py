from .apply_augmentations import Augment

import numpy as np


def crop_img(img_numpy, crop_size, crop):
    """

    Args:
        img_numpy:
        crop_size:
        crop:

    Returns:

    """
    if crop_size[0] == 0:
        return img_numpy
    slices_crop, w_crop, h_crop = crop

    dim1, dim2, dim3 = crop_size
    inp_img_dim = img_numpy.ndim

    assert inp_img_dim >= 3
    if img_numpy.ndim== 3:
        full_dim1, full_dim2, full_dim3 = img_numpy.shape
    elif img_numpy.ndim == 4:
        _, full_dim1, full_dim2, full_dim3 = img_numpy.shape
        img_numpy = img_numpy[0, ...]


    img_numpy = img_numpy[slices_crop:slices_crop + dim1, w_crop:w_crop + dim2,
                    h_crop:h_crop + dim3]

    if inp_img_dim == 4:
        return img_numpy.unsqueeze(0)

    return img_numpy


def find_random_crop_dim(full_vol_dim, crop_size):
    """

    Args:
        full_vol_dim:
        crop_size:

    Returns:

    """
    assert full_vol_dim[0] >= crop_size[0], "crop size is too big"
    assert full_vol_dim[1] >= crop_size[1], "crop size is too big"
    assert full_vol_dim[2] >= crop_size[2], "crop size is too big"

    if full_vol_dim[0] == crop_size[0]:
        slices = crop_size[0]
    else:
        slices = np.random.randint(full_vol_dim[0] - crop_size[0])

    if full_vol_dim[1] == crop_size[1]:
        w_crop = crop_size[1]
    else:
        w_crop = np.random.randint(full_vol_dim[1] - crop_size[1])

    if full_vol_dim[2] == crop_size[2]:
        h_crop = crop_size[2]
    else:
        h_crop = np.random.randint(full_vol_dim[2] - crop_size[2])

    return (slices, w_crop, h_crop)


class RandomCrop(Augment):
    def __init__(self, modality_keys, apply_to_label=True, full_vol_dim=None, crop_size=None):
        super(RandomCrop, self).__init__(modality_keys, apply_to_label)
        self.crop_size = crop_size
        self.full_vol_dim = full_vol_dim

    def __call__(self, data):

        crop = find_random_crop_dim(self.full_vol_dim, self.crop_size)
        for key in self.modality_keys:
            if key is not 'label':
                data[key] = crop_img(data[key], self.crop_size, crop)
        if self.apply_to_label:
            data['label'] = crop_img(data['label'], self.crop_size, crop)
        return data
