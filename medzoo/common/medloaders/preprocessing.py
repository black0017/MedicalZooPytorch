import numpy as np
import scipy.ndimage  as ndimage
import torch


def transform_coordinate_space(modality_1, modality_2):
    """
    Accepts nifty objects
    Transfers coordinate space from modality_2 to modality_1
    """
    aff_t1 = modality_1.affine
    aff_t2 = modality_2.affine
    inv_af_2 = np.linalg.inv(aff_t2)

    out_shape = modality_1.get_fdata().shape

    # desired transformation
    T = inv_af_2.dot(aff_t1)
    transformed = ndimage.affine_transform(modality_2.get_fdata(), T, output_shape=out_shape)

    return transformed


def medical_image_transform(img_tensor, type=None,
                            normalization="full_volume_mean",
                            norm_values=(0., 1., 1., 0.)):
    """

    Args:
        img_tensor:
        type:
        normalization:
        norm_values:

    Returns:

    """
    MEAN, STD, MAX, MIN = norm_values
    # Numpy-based transformations/augmentations here

    if type != 'label':
        MEAN, STD = img_tensor.mean(), img_tensor.std()
        MAX, MIN = img_tensor.max(), img_tensor.min()

    if type != "label":
        img_tensor = normalize_intensity(img_tensor, normalization=normalization, norm_values=(MEAN, STD, MAX, MIN))

    return img_tensor


def crop_img(img_tensor, crop_size, crop):
    """

    Args:
        img_tensor:
        crop_size:
        crop:

    Returns:

    """
    if crop_size[0] == 0:
        return img_tensor
    slices_crop, w_crop, h_crop = crop
    dim1, dim2, dim3 = crop_size
    inp_img_dim = img_tensor.dim()
    assert inp_img_dim >= 3
    if img_tensor.dim() == 3:
        full_dim1, full_dim2, full_dim3 = img_tensor.shape
    elif img_tensor.dim() == 4:
        _, full_dim1, full_dim2, full_dim3 = img_tensor.shape
        img_tensor = img_tensor[0, ...]

    if full_dim1 == dim1:
        img_tensor = img_tensor[:, w_crop:w_crop + dim2,
                     h_crop:h_crop + dim3]
    elif full_dim2 == dim2:
        img_tensor = img_tensor[slices_crop:slices_crop + dim1, :,
                     h_crop:h_crop + dim3]
    elif full_dim3 == dim3:
        img_tensor = img_tensor[slices_crop:slices_crop + dim1, w_crop:w_crop + dim2, :]
    else:
        img_tensor = img_tensor[slices_crop:slices_crop + dim1, w_crop:w_crop + dim2,
                     h_crop:h_crop + dim3]

    if inp_img_dim == 4:
        return img_tensor.unsqueeze(0)

    return img_tensor


def rescale_data_volume(img_numpy, out_dim):
    """Resize the 3d numpy array to the dim size

    Args:
        out_dim is the new 3d tuple
    """
    depth, height, width = img_numpy.shape
    scale = [out_dim[0] * 1.0 / depth, out_dim[1] * 1.0 / height, out_dim[2] * 1.0 / width]
    return ndimage.interpolation.zoom(img_numpy, scale, order=0)


def rescale_volume_iso(img_numpy, scale, order=0):
    """Isometric rescaling

    Args:
        scale: the new scale (float)
        order: from 0 to 5
    """
    return ndimage.interpolation.zoom(img_numpy, scale, order=order)


def normalize_intensity(img_tensor, normalization="full_volume_mean"):
    """Accepts an image tensor and normalizes it

    Args:
        normalization: choices = "max", "mean" , type=str
    """
    MEAN, STD = img_tensor.mean(), img_tensor.std()
    MAX, MIN = img_tensor.max(), img_tensor.min()
    # Non-zero voxel mean/std
    if normalization == "mean":
        mask = img_tensor.ne(0.0)
        desired = img_tensor[mask]
        mean_val, std_val = desired.mean(), desired.std()
        img_tensor = (img_tensor - mean_val) / std_val
    elif normalization == 'brats':
        normalized_tensor = (img_tensor.clone() - MEAN) / STD
        final_tensor = torch.where(img_tensor == 0., img_tensor, normalized_tensor)
        final_tensor = 100.0 * ((final_tensor.clone() - MIN) / (MAX - MIN)) + 10.0
        img_tensor = torch.where(img_tensor == 0., img_tensor, final_tensor)
    # voxel mean/std, zero intensity voxels included
    elif normalization == 'full_volume_mean':
        img_tensor = (img_tensor.clone() - MEAN) / STD
    elif normalization == 'max_min' or normalization == 'max':
        img_tensor = (img_tensor - MIN) / ((MAX - MIN))

    return img_tensor


def percentile_clip(img_numpy, min_val=0.1, max_val=99.8):
    """Intensity normalization based on percentile
    Clips the range based on the quarile values.

    Args:
    min_val: should be in the range [0,100]
    max_val: should be in the range [0,100]

    Returns:
        intesity normalized image
    """
    low = np.percentile(img_numpy, min_val)
    high = np.percentile(img_numpy, max_val)

    img_numpy[img_numpy < low] = low
    img_numpy[img_numpy > high] = high
    return img_numpy


def clip_range_intensity(img_numpy, min_val, max_val):
    return np.clip(img_numpy, min_val, max_val)
