import matplotlib.pyplot as plt
import os
import numpy as np


def show_mid_slice(img_numpy, return_views=False):
    """
    Accepts an 3D numpy array and shows median slices in all three planes
    :param img_numpy:
    """
    assert img_numpy.ndim == 3, "please provide a 3d numpy image"
    n_i, n_j, n_k = img_numpy.shape

    # saggital
    center_i1 = int((n_i - 1) / 2)
    # transversecreate_2d_views
    center_j1 = int((n_j - 1) / 2)
    # axial slice
    center_k1 = int((n_k - 1) / 2)

    if return_views == False:
        show_slices([img_numpy[center_i1, :, :],
                     img_numpy[:, center_j1, :],
                     img_numpy[:, :, center_k1]])
    else:
        return (img_numpy[center_i1, :, :],
                img_numpy[:, center_j1, :],
                img_numpy[:, :, center_k1])


def show_slices(slices):
    """
    Function to display a row of image slices
    Input is a list of numpy 2D image slices
    """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")


# TODO test and add medical writer
def create_2d_views(predictions, segment_map, path_to_save):
    """
    Comparative 2d vizualization of median slices:
    axial, saggital and transpose. Save to png file and to tensorboard
    :param predictions:
    :param segment_map:
    :param epoch:
    :param writer:
    :param path_to_save:
    :return:
    """
    # todo scale to range [0,255 for tensor borad] ??????
    segment_pred = seg_map_vizualization_iseg(predictions)

    s1, s2, s3 = show_mid_slice(segment_pred, return_views=True)
    p1, p2, p3 = show_mid_slice(segment_map, return_views=True)

    assert s1.shape == p1.shape
    assert s2.shape == p2.shape
    assert s3.shape == p3.shape

    list_vol = [s1, p1, s2, p2, s3, p3]
    rows, columns = 3, 2
    figure = plt.figure(figsize=(16, 16))
    for i in range(len(list_vol)):
        figure.add_subplot(rows, columns, i + 1)
        # plt.imshow(list_vol[i], cmap='gray')
        plt.imshow(list_vol[i])

    plt.savefig(path_to_save)
    print("DONEEEEEEEEEEEEEEEE 2D views production....")
    # TODO test in base class writer
    # writer.add_figure('Images/all_2d_views', figure, epoch)
    # writer.add_image('Images/pred_view_1', s1, epoch, dataformats='HW')
    # writer.add_image('Images/pred_view_2', s2, epoch, dataformats='HW')
    # writer.add_image('Images/pred_view_3', s3, epoch, dataformats='HW')
    # TODO save image pairs
    # a1 = torch.stack((s1, p1)).long()
    # writer.add_images('view_1', a1, epoch, dataformats='NHWC' )


def seg_map_vizualization_iseg(segmentation_map):
    # visual labels of ISEG-2017
    label_values = [0, 10, 150, 250]
    for c, j in enumerate(label_values):
        segmentation_map[segmentation_map == c] = j
    return segmentation_map


# not used right now ???
def plot_segm(segm, ground_truth, plots_dir='.'):
    """
    Saves predicted and ground truth segmentation into a PNG files (one per channel).
    :param segm: 4D ndarray (CDHW)
    :param ground_truth: 4D ndarray (CDHW)
    :param plots_dir: directory where to save the plots
    """
    import uuid
    assert segm.ndim == 4
    if ground_truth.ndim == 3:
        stacked = [ground_truth for _ in range(segm.shape[0])]
        ground_truth = np.stack(stacked)

    assert ground_truth.ndim == 4

    f, axarr = plt.subplots(1, 2)

    for seg, gt in zip(segm, ground_truth):
        mid_z = seg.shape[0] // 2

        axarr[0].imshow(seg[mid_z], cmap='prism')
        axarr[0].set_title('Predicted segmentation')

        axarr[1].imshow(gt[mid_z], cmap='prism')
        axarr[1].set_title('Ground truth segmentation')

        file_name = f'segm_{str(uuid.uuid4())[:8]}.png'
        plt.savefig(os.path.join(plots_dir, file_name))


def overlap_2d_image():
    B, C, W, H = 2, 3, 1024, 1024
    x = torch.randn(B, C, H, W)

    kernel_size = 128
    stride = 64
    patches = x.unfold(3, kernel_size, stride).unfold(2, kernel_size, stride)
    #print('patches shape ', patches.shape)  # [B, C, nb_patches_h, nb_patches_w, kernel_size, kernel_size]

    # perform the operations on each patch
    # ...

    # reshape output to match F.fold input
    patches = patches.contiguous().view(B, C, -1, kernel_size * kernel_size)
    #print(patches.shape)  # [B, C, nb_patches_all, kernel_size*kernel_size]
    patches = patches.permute(0, 1, 3, 2)
    #print(patches.shape)  # [B, C, kernel_size*kernel_size, nb_patches_all]
    patches = patches.contiguous().view(B, C * kernel_size * kernel_size, -1)
    #print(patches.shape)  # [B, C*prod(kernel_size), L] as expected by Fold
    # https://pytorch.org/docs/stable/nn.html#torch.nn.Fold

    output = F.fold(
        patches, output_size=(H, W), kernel_size=kernel_size, stride=stride)
    #print(output.shape)  # [B, C, H, W]
