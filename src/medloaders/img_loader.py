import nibabel as nib
import torch
import numpy as np


def load_medical_image(path, crop_size=(0, 0, 0), crop=(0, 0, 0), type=None, normalization="mean", viz3d=False):
    slices_crop, w_crop, h_crop = crop
    img = nib.load(path)
    img_np = np.squeeze(img.get_fdata())
    img_tensor = torch.from_numpy(img_np).float()  # slice , width, height

    if not viz3d:
        img_tensor = img_tensor[slices_crop:slices_crop + crop_size[0], w_crop:w_crop + crop_size[1],
                     h_crop:h_crop + crop_size[2]]

    if type != "label":
        if normalization == "mean":
            mask = img_tensor.ne(0.0)
            desired = img_tensor[mask]
            mean_val, std_val = desired.mean(), desired.std()
            img_tensor = (img_tensor - mean_val) / std_val
        else:
            img_tensor = img_tensor / 255.0

        return img_tensor.unsqueeze(0)
    else:
        return img_tensor


def load_affine_matrix(path):
    img = nib.load(path)
    return img.affine
