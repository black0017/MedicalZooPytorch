import nibabel as nib
import numpy as np
import torch
from PIL import Image
from nibabel.processing import resample_to_output
from scipy import ndimage

"""
concentrate all pre-processing here here
"""


def load_medical_image(path, crop_size=(0, 0, 0), crop=(0, 0, 0), type=None, normalization="mean", resample=None,
                       viz3d=False, to_canonical=False, rescale=None):
    img_nii = nib.load(path)

    # Medical img proccesing pipiline functions here
    if to_canonical:
        img_nii = nib.as_closest_canonical(img_nii)

    if resample is not None:
        img_nii = resample_to_output(img_nii, voxel_sizes=resample)

    img_np = np.squeeze(img_nii.get_fdata(dtype=np.float32))
    if viz3d:
        return torch.from_numpy(img_np)

    # Numpy-based transformations/augmentations here
    if rescale is not None:
        rescale_data_volume(img_np, rescale)

    if crop_size[0] != 0:
        slices_crop, w_crop, h_crop = crop
        img_np = img_np[slices_crop:slices_crop + crop_size[0], w_crop:w_crop + crop_size[1],
                 h_crop:h_crop + crop_size[2]]

    # Tensor proccesing here
    img_tensor = torch.from_numpy(img_np)

    if type != "label":
        img_tensor = normalize_intensity(img_tensor, normalization=normalization)
    return img_tensor


def load_affine_matrix(path):
    """
    Reads an path to nifti file and returns the affine matrix as numpy array 4x4
    """
    img = nib.load(path)
    return img.affine


def load_2d_image(img_path, resize_dim=0, type='RGB'):
    image = Image.open(img_path)
    if type == 'RGB':
        image = image.convert(type)
    if resize_dim != 0:
        image = image.resize(resize_dim)
    pix = np.array(image)
    return pix


def rescale_data_volume(img_numpy, out_dim):
    """
    Resize the 3d numpy array to the dim size
    :param out_dim is the new 3d tuple
    """
    depth, height, width = img_numpy.shape
    scale = [out_dim[0] * 1.0 / depth, out_dim[1] * 1.0 / height, out_dim[2] * 1.0 / width]
    return ndimage.interpolation.zoom(img_numpy, scale, order=0)


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


def normalize_intensity(img_tensor, normalization="mean"):
    """
    Accepts an image tensor and normalizes it
    :param normalization: choices = "max", "mean" , type=str
    """
    if normalization == "mean":
        mask = img_tensor.ne(0.0)
        desired = img_tensor[mask]
        mean_val, std_val = desired.mean(), desired.std()
        img_tensor = (img_tensor - mean_val) / std_val
    elif normalization == "max":
        max_val, _ = torch.max(img_tensor)
        img_tensor = img_tensor / max_val
    return img_tensor.unsqueeze(0)


## todo percentiles

def clip_range(img_numpy):
    """
    Cut off outliers that are related to detected black in the image (the air area)
    """
    # Todo median value!
    zero_value = (img_numpy[0, 0, 0] + img_numpy[-1, 0, 0] + img_numpy[0, -1, 0] + \
                  img_numpy[0, 0, -1] + img_numpy[-1, -1, -1] + img_numpy[-1, -1, 0] \
                  + img_numpy[0, -1, -1] + img_numpy[-1, 0, -1]) / 8.0
    non_zeros_idx = np.where(img_numpy >= zero_value)
    [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
    [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)
    y = img_numpy[min_z:max_z, min_h:max_h, min_w:max_w]
    return y
