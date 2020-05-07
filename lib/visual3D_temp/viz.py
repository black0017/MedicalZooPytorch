import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import os

from lib.utils.general import prepare_input
from .viz_2d import *


def visualize_3D_no_overlap_new(args, full_volume, affine, model, epoch, dim):
    """
    this function will produce NON-overlaping  sub-volumes prediction
    that produces full 3d medical image
    compare some slices with ground truth
    :param full_volume: t1, t2, segment
    :param dim: (d1,d2,d3))
    :return: 3d reconstructed volume
    """
    classes = args.classes
    modalities, slices, height, width = full_volume.shape
    full_volume_dim = (slices, height, width)

    print("full volume dim=", full_volume_dim, 'crop dim', dim)
    desired_dim = find_crop_dims(full_volume_dim, dim)
    print("Inference dims=", desired_dim)

    input_sub_volumes, segment_map = create_3d_subvol(full_volume, desired_dim)
    print(input_sub_volumes.shape, segment_map.shape)

    sub_volumes = input_sub_volumes.shape[0]
    predictions = []

    for i in range(sub_volumes):
        input_tensor = input_sub_volumes[i, ...].unsqueeze(0)
        predictions.append(model.inference(input_tensor))

    predictions = torch.stack(predictions)
    # project back to full volume
    full_vol_predictions = predictions.view(classes, slices, height, width)
    print("Inference complete", full_vol_predictions.shape)

    # arg max to get the labels in full 3d volume
    _, indices = full_vol_predictions.max(dim=0)
    full_vol_predictions = indices

    print("Class indexed prediction shape", full_vol_predictions.shape, "GT", segment_map.shape)

    # TODO TEST...................
    save_path_2d_fig = args.save + '/' + 'epoch__' + str(epoch).zfill(4) + '.png'
    create_2d_views(full_vol_predictions, segment_map, save_path_2d_fig)

    save_path = args.save + '/Pred_volume_epoch_' + str(epoch)
    save_3d_vol(full_vol_predictions.numpy(), affine, save_path)


# TODO TEST
def create_3d_subvol(full_volume, dim):
    list_modalities = []

    modalities, slices, height, width = full_volume.shape

    full_vol_size = tuple((slices, height, width))
    dim = find_crop_dims(full_vol_size, dim)
    for i in range(modalities):
        TARGET_VOL = modalities - 1

        if i != TARGET_VOL:
            img_tensor = full_volume[i, ...]
            img = grid_sampler_sub_volume_reshape(img_tensor, dim)
            list_modalities.append(img)
        else:
            target = full_volume[i, ...]

    input_tensor = torch.stack(list_modalities, dim=1)

    return input_tensor, target


def grid_sampler_sub_volume_reshape(tensor, dim):
    return tensor.view(-1, dim[0], dim[1], dim[2])


def find_crop_dims(full_size, mini_dim, adjust_dimension=2):
    a, b, c = full_size
    d, e, f = mini_dim

    voxels = a * b * c
    subvoxels = d * e * f

    if voxels % subvoxels == 0:
        return mini_dim

    static_voxels = mini_dim[adjust_dimension - 1] * mini_dim[adjust_dimension - 2]
    print(static_voxels)
    if voxels % static_voxels == 0:
        temp = int(voxels / static_voxels)
        print("temp=", temp)
        mini_dim_slice = mini_dim[adjust_dimension]
        step = 1
        while True:
            slice_dim1 = temp % (mini_dim_slice - step)
            slice_dim2 = temp % (mini_dim_slice + step)
            if slice_dim1 == 0:
                slice_dim = int(mini_dim_slice - step)
                break
            elif slice_dim2 == 0:
                slice_dim = int(temp / (mini_dim_slice + step))
                break
            else:
                step += 1
        return (d, e, slice_dim)

    full_slice = full_size[adjust_dimension]

    return tuple(desired_dim)


# Todo  test!
def save_3d_vol(predictions, affine, save_path):
    pred_nifti_img = nib.Nifti1Image(predictions, affine)
    pred_nifti_img.header["qform_code"] = 1
    pred_nifti_img.header['sform_code'] = 0
    nib.save(pred_nifti_img, save_path + '.nii.gz')
    print('3D vol saved')
    # alternativly  pred_nifti_img.tofilename(str(save_path))
