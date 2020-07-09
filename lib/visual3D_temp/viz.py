import math

import nibabel as nib
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from .viz_2d import *
from lib.medloaders.medical_image_process import medical_image_transform, pad_medical_image

from lib.losses3D import DiceLoss
def test_padding():
    x = torch.randn(1, 144, 192, 256)
    kc, kh, kw = 32, 32, 32  # kernel size
    dc, dh, dw = 32, 32, 32  # stride
    # Pad to multiples of 32
    x = F.pad(x, (x.size(3) % kw // 2, x.size(3) % kw // 2,
                  x.size(2) % kh // 2, x.size(2) % kh // 2,
                  x.size(1) % kc // 2, x.size(1) % kc // 2))
    print(x.shape)
    patches = x.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
    unfold_shape = patches.size()
    print(unfold_shape)
    patches = patches.contiguous().view(-1, kc, kh, kw)
    print(patches.shape)

    # Reshape back
    patches_orig = patches.view(unfold_shape)
    output_c = unfold_shape[1] * unfold_shape[4]
    output_h = unfold_shape[2] * unfold_shape[5]
    output_w = unfold_shape[3] * unfold_shape[6]
    patches_orig = patches_orig.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    patches_orig = patches_orig.view(1, output_c, output_h, output_w)

    # Check for equality
    print((patches_orig == x[:, :output_c, :output_h, :output_w]).all())


def non_overlap_padding2(args, full_volume, model, criterion, kernel_dim=(32, 32, 32)):
    x = full_volume[:-1, ...].detach().cpu()

    target = full_volume[-1, ...].unsqueeze(0).detach().cpu()

    if args.cuda:
        model = model.cuda()
        # full_volume = full_volume.cuda()
        # x = x.cuda()
        # target = target.cuda()

    modalities, D, H, W = x.shape
    kc, kh, kw = kernel_dim
    dc, dh, dw = kernel_dim[0], kernel_dim[0], kernel_dim[0]  # stride
    # Pad to multiples of kernel_dim
    x, a = pad_medical_image(x, kernel_dim)
    new_target, _ = pad_medical_image(target, kernel_dim)

    assert x.size(3) % kw == 0
    assert x.size(2) % kh == 0
    assert x.size(1) % kc == 0
    patches = x.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
    unfold_shape = list(patches.size())
    new_target = new_target.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw).contiguous().view(-1, 1, kc, kh, kw)
    patches = patches.contiguous().view(-1, modalities, kc, kh, kw)

    ## TODO torch stack
    # with torch.no_grad():
    #     output = model.inference(patches)

    avg_loss = 0.0
    number_of_volumes = patches.shape[0]
    predictions = []
    ch_scores = np.zeros((3,))
    with torch.no_grad():
        for i in range(number_of_volumes):
            input_tensor = patches[i, ...].unsqueeze(0)
            o = model.inference(input_tensor.cuda())
            predictions.append(o)

            l, c = criterion(o.cuda(), new_target[i, ...].cuda())
            # print('Inference train subvolume skip air {} {} '.format(l.item(),c))
            criterion2 = DiceLoss(classes=4, skip_background_class=False)
            # l1, c2 = criterion2(o.cuda(), new_target[i, ...].cuda())
            # print('Inference train subvolume {} {} '.format(l1.item(), c2))
            # print('non zero target voxels {}'.format((new_target[i,...] != 0).sum() ))

            # ch_scores +=c
            avg_loss += l.item()
    output = torch.stack(predictions, dim=0).squeeze(1).detach()
    # print(output.shape)
    N, Classes, _, _, _ = output.shape
    # Reshape backlist
    output_unfold_shape = unfold_shape[1:]
    output_unfold_shape.insert(0, Classes)
    # print(output_unfold_shape)
    output = output.view(output_unfold_shape)

    output_c = output_unfold_shape[1] * output_unfold_shape[4]
    output_h = output_unfold_shape[2] * output_unfold_shape[5]
    output_w = output_unfold_shape[3] * output_unfold_shape[6]
    output = output.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    output = output.view(-1, output_c, output_h, output_w)
    print(output.shape)

    y = output[:, a[4]:output_c - a[5], a[2]:output_h - a[3], a[0]:output_w - a[1]]

    # print(y.shape)

    loss_dice, per_ch_score = criterion(y.unsqueeze(0).cuda(), target.cuda())
    # print(ch_scores / i)
    print("INFERENCE AVERAGE {} {}".format(avg_loss / number_of_volumes, ch_scores / number_of_volumes))
    print("FULL VOLUME DICE LOSS {} {}".format(loss_dice, per_ch_score))
    # print(y.shape,target.shape)

    return y


def non_overlap_padding_v1(args, full_volume, model, criterion, kernel_dim=(32, 32, 32)):
    if args.cuda:
        model = model.cuda()
        full_volume = full_volume.cuda()

    x = full_volume[0, ...].unsqueeze(0).detach()
    # print(x[0,0,0,0],x[0,1,1,1],x[0,2,1,1])

    target = full_volume[-1, ...].unsqueeze(0).detach()
    # plt.imshow(x.cpu().numpy()[0,...,20])
    # plt.show()
    # print(x.shape)

    x, a = pad_medical_image(x, kernel_dim)
    # print(x.shape)
    # plt.imshow(x.cpu().numpy()[0,...,20])
    # plt.show()
    modalities, D, H, W = x.shape
    kc, kh, kw = kernel_dim
    dc, dh, dw = kernel_dim  # stride

    # print('padded shape ', x.shape)
    assert x.size(3) % kw == 0
    assert x.size(2) % kh == 0
    assert x.size(1) % kc == 0
    patches = x.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
    unfold_shape = list(patches.size())

    patches = patches.contiguous().view(-1, modalities, kc, kh, kw)

    ## TODO torch stack
    # with torch.no_grad():
    #     output = model.inference(patches)
    number_of_volumes = patches.shape[0]
    predictions = []
    with torch.no_grad():
        for i in range(number_of_volumes):
            input_tensor = patches[i, ...].unsqueeze(0)
            o = model.inference(input_tensor)
            predictions.append(o)
            # print(o.shape, target.shape)
    output = torch.stack(predictions, dim=0).squeeze(1).detach()
    # print(output.shape)
    N, Classes, _, _, _ = output.shape
    # Reshape backlist
    output_unfold_shape = unfold_shape[1:]
    output_unfold_shape.insert(0, Classes)
    # print(output_unfold_shape)
    output = output.view(output_unfold_shape)

    output_c = output_unfold_shape[1] * output_unfold_shape[4]
    output_h = output_unfold_shape[2] * output_unfold_shape[5]
    output_w = output_unfold_shape[3] * output_unfold_shape[6]
    output = output.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    output = output.view(-1, output_c, output_h, output_w)
    # print(output.shape)

    y = output[:, a[4]:output_c - a[5], a[2]:output_h - a[3], a[0]:output_w - a[1]]

    # print(y.shape,target.shape)

    loss_dice, per_ch_score = criterion(y.unsqueeze(0).cuda(), target.cuda())

    print("INFERENCE DICE LOSS {} {}".format(loss_dice.item(), per_ch_score))
    return y


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
