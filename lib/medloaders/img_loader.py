import nibabel as nib
import torch
import numpy as np
from nibabel.processing import resample_to_output
from PIL import Image


def load_medical_image(path, crop_size=(0, 0, 0), crop=(0, 0, 0), type=None, normalization="mean", resample=None,
                       viz3d=False, to_canonical=False):
    slices_crop, w_crop, h_crop = crop
    img_nii = nib.load(path)
    img_np = np.squeeze(img_nii.get_fdata(dtype=np.float32))


    if viz3d:
        return torch.from_numpy(img_np)

    if to_canonical:
        img_nii = nib.as_closest_canonical(img_nii)
        img_np = img_nii.get_fdata(dtype=np.float32)

    if crop_size[0] != 0:
        img_np = img_np[slices_crop:slices_crop + crop_size[0], w_crop:w_crop + crop_size[1],
                 h_crop:h_crop + crop_size[2]]

    if resample is not None:
        img_nii = resample_to_output(img_nii, voxel_sizes=resample)
        img_np = np.squeeze(img_nii.get_fdata(dtype=np.float32))

    img_tensor = torch.from_numpy(img_np)  # slice , width, height
    #print('final tensor shape', img_tensor.shape)

    if type != "label":
        img_tensor = normalize(img_tensor, normalization=normalization)
    return img_tensor


def normalize(img_tensor, normalization="max"):
    if normalization == "mean":
        mask = img_tensor.ne(0.0)
        desired = img_tensor[mask]
        mean_val, std_val = desired.mean(), desired.std()
        img_tensor = (img_tensor - mean_val) / std_val
    elif normalization == "max":
        max_val, _ = torch.max(img_tensor)
        img_tensor = img_tensor / max_val
    return img_tensor.unsqueeze(0)


def load_affine_matrix(path):
    img = nib.load(path)
    return img.affine


def load_2d_image(img_path, resize_dim=0, type='RGB'):
    image = Image.open(img_path).convert(type)
    if resize_dim != 0:
        image = image.resize(resize_dim)
    return image
