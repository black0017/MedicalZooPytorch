import nibabel as nib
import numpy as np
import torch
from PIL import Image
from nibabel.processing import resample_to_output
from scipy import ndimage

"""
concentrate all pre-processing here here
"""


def load_medical_image(path, type=None, resample=None,
                       viz3d=False, to_canonical=False, rescale=None, normalization='full_volume_mean',
                       clip_intenisty=True, crop_size=(0, 0, 0), crop=(0, 0, 0), ):
    img_nii = nib.load(path)

    if to_canonical:
        img_nii = nib.as_closest_canonical(img_nii)

    if resample is not None:
        img_nii = resample_to_output(img_nii, voxel_sizes=resample)

    img_np = np.squeeze(img_nii.get_fdata(dtype=np.float32))

    if viz3d:
        return torch.from_numpy(img_np)

    # 1. Intensity outlier clipping
    if clip_intenisty and type != "label":
        img_np = percentile_clip(img_np)

    # 2. Rescale to specified output shape
    if rescale is not None:
        rescale_data_volume(img_np, rescale)

    # 3. intensity normalization
    img_tensor = torch.from_numpy(img_np)

    MEAN, STD, MAX, MIN = 0., 1., 1., 0.
    if type != 'label':
        MEAN, STD = img_tensor.mean(), img_tensor.std()
        MAX, MIN = img_tensor.max(), img_tensor.min()
    if type != "label":
        img_tensor = normalize_intensity(img_tensor, normalization=normalization, norm_values=(MEAN, STD, MAX, MIN))
    img_tensor = crop_img(img_tensor, crop_size, crop)
    return img_tensor


def medical_image_transform(img_tensor, type=None,
                            normalization="full_volume_mean",
                            norm_values=(0., 1., 1., 0.)):
    MEAN, STD, MAX, MIN = norm_values
    # Numpy-based transformations/augmentations here

    if type != 'label':
        MEAN, STD = img_tensor.mean(), img_tensor.std()
        MAX, MIN = img_tensor.max(), img_tensor.min()

    if type != "label":
        img_tensor = normalize_intensity(img_tensor, normalization=normalization, norm_values=(MEAN, STD, MAX, MIN))

    return img_tensor


def crop_img(img_tensor, crop_size, crop):
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


def normalize_intensity(img_tensor, normalization="full_volume_mean", norm_values=(0, 1, 1, 0)):
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
    elif normalization == 'brats':
        # print(norm_values)
        normalized_tensor = (img_tensor.clone() - norm_values[0]) / norm_values[1]
        final_tensor = torch.where(img_tensor == 0., img_tensor, normalized_tensor)
        final_tensor = 100.0 * ((final_tensor.clone() - norm_values[3]) / (norm_values[2] - norm_values[3])) + 10.0
        x = torch.where(img_tensor == 0., img_tensor, final_tensor)
        return x

    elif normalization == 'full_volume_mean':
        img_tensor = (img_tensor.clone() - norm_values[0]) / norm_values[1]

    elif normalization == 'max_min':
        img_tensor = (img_tensor - norm_values[3]) / ((norm_values[2] - norm_values[3]))

    elif normalization == None:
        img_tensor = img_tensor
    return img_tensor


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


def percentile_clip(img_numpy, min_val=0.1, max_val=99.8):
    """
    Intensity normalization based on percentile
    Clips the range based on the quarile values.
    :param min_val: should be in the range [0,100]
    :param max_val: should be in the range [0,100]
    :return: intesity normalized image
    """
    low = np.percentile(img_numpy, min_val)
    high = np.percentile(img_numpy, max_val)

    img_numpy[img_numpy < low] = low
    img_numpy[img_numpy > high] = high
    return img_numpy
