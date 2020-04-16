import nibabel as nib
import torch
import numpy as np
from nibabel.processing import resample_to_output
from PIL import Image
from scipy import ndimage

"""
concentrate all preprocessing here here
"""


def load_medical_image(path, crop_size=(0, 0, 0), crop=(0, 0, 0), type=None, normalization="mean", resample=None,
                       viz3d=False, to_canonical=False, random_rotate3D=None, rescale=None):
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

    if random_rotate3D is not None:
        img_np = random_rotate3D(img_np, -random_rotate3D, random_rotate3D)

    if crop_size[0] != 0:
        slices_crop, w_crop, h_crop = crop
        img_np = img_np[slices_crop:slices_crop + crop_size[0], w_crop:w_crop + crop_size[1],
                 h_crop:h_crop + crop_size[2]]

    # Tensor proccesing here
    img_tensor = torch.from_numpy(img_np)  # slice , width, height
    # print('final tensor shape', img_tensor.shape)

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
    image = Image.open(img_path).convert(type)
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


def random_rotate3D(img_numpy, min_angle, max_angle):
    """
    Returns a random rotated array in the same shape
    :param img_numpy: 3D numpy array
    :param min_angle: in degrees
    :param max_angle: in degrees
    :return:
    """
    assert img_numpy.ndim == 3, "provide a 3d numpy array"
    assert min_angle < max_angle, "min should be less than max val"
    assert min_angle > -360 or max_angle < 360
    all_axes = [(1, 0), (1, 2), (0, 2)]
    angle = np.random.randint(low=min_angle, high=max_angle+1)
    axes_random_id = np.random.randint(low=0, high=len(all_axes))
    axes = all_axes[axes_random_id]

    return ndimage.rotate(img_numpy, angle, axes=axes)


# todo TEST
def clip_range(img_numpy):
    """
    Cut off the invalid area
    """
    zero_value = img_numpy[0, 0, 0]

    non_zeros_idx = np.where(img_numpy != zero_value)
    print(non_zeros_idx)
    print(zero_value)
    print("global min and max")
    print(np.max(img_numpy))
    print(np.min(img_numpy))

    [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
    [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)
    print("max and min per axis")
    print(max_z, max_h, max_w)
    print(min_z, min_h, min_w)
    y = img_numpy[min_z:max_z, min_h:max_h, min_w:max_w]
    print('new min max values')
    print(np.max(y))
    print(np.min(y))
    return y


def random_crop_to_labels(data, label):
    """
    Random center crop
    """
    target_indexs = np.where(label > 0)
    [img_d, img_h, img_w] = data.shape
    [max_D, max_H, max_W] = np.max(np.array(target_indexs), axis=1)
    [min_D, min_H, min_W] = np.min(np.array(target_indexs), axis=1)
    [target_depth, target_height, target_width] = np.array([max_D, max_H, max_W]) - np.array([min_D, min_H, min_W])
    Z_min = int((min_D - target_depth * 1.0 / 2) * np.random_sample())
    Y_min = int((min_H - target_height * 1.0 / 2) * np.random_sample())
    X_min = int((min_W - target_width * 1.0 / 2) * np.random_sample())

    Z_max = int(img_d - ((img_d - (max_D + target_depth * 1.0 / 2)) * np.random_sample()))
    Y_max = int(img_h - ((img_h - (max_H + target_height * 1.0 / 2)) * np.random_sample()))
    X_max = int(img_w - ((img_w - (max_W + target_width * 1.0 / 2)) * np.random_sample()))

    Z_min = int(np.max([0, Z_min]))
    Y_min = int(np.max([0, Y_min]))
    X_min = int(np.max([0, X_min]))

    Z_max = int(np.min([img_d, Z_max]))
    Y_max = int(np.min([img_h, Y_max]))
    X_max = int(np.min([img_w, X_max]))

    return data[Z_min: Z_max, Y_min: Y_max, X_min: X_max]