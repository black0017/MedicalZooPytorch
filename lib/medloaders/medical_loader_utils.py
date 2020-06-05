from lib.medloaders import medical_image_process as img_loader
from lib.visual3D_temp import *


def get_viz_set(*ls, dataset_name, test_subject=0, save=False, sub_vol_path=None):
    """
    Returns total 3d input volumes (t1 and t2 or more) and segmentation maps
    3d total vol shape : torch.Size([1, 144, 192, 256])
    """
    modalities = len(ls)
    total_volumes = []

    for i in range(modalities):
        path_img = ls[i][test_subject]

        img_tensor = img_loader.load_medical_image(path_img, viz3d=True)
        if i == modalities - 1:
            img_tensor = fix_seg_map(img_tensor, dataset=dataset_name)

        total_volumes.append(img_tensor)

    if save:
        total_subvolumes = total_volumes[0].shape[0]
        for i in range(total_subvolumes):
            filename = sub_vol_path + 'id_' + str(test_subject) + '_VIZ_' + str(i) + '_modality_'
            for j in range(modalities):
                filename = filename + str(j) + '.npy'
                np.save(filename, total_volumes[j][i])
    else:
        return torch.stack(total_volumes, dim=0)


def fix_seg_map(segmentation_map, dataset="iseg2017"):
    if dataset == "iseg2017" or dataset == "iseg2019":
        label_values = [0, 10, 150, 250]
        for c, j in enumerate(label_values):
            segmentation_map[segmentation_map == j] = c

    elif dataset == "brats2018" or dataset == "brats2019" or dataset == "brats2020":
        ED = 2
        NCR = 1
        NET_NCR = 1
        ET = 3
        # print('dsdsdsdsd')
        segmentation_map[segmentation_map == 1] = NET_NCR
        segmentation_map[segmentation_map == 2] = ED
        segmentation_map[segmentation_map == 3] = 3
        segmentation_map[segmentation_map == 4] = 3
        segmentation_map[segmentation_map >= 4] = 3
    elif dataset == "mrbrains4":
        GM = 1
        WM = 2
        CSF = 3
        segmentation_map[segmentation_map == 1] = GM
        segmentation_map[segmentation_map == 2] = GM
        segmentation_map[segmentation_map == 3] = WM
        segmentation_map[segmentation_map == 4] = WM
        segmentation_map[segmentation_map == 5] = CSF
        segmentation_map[segmentation_map == 6] = CSF
    return segmentation_map


def create_sub_volumes(*ls, dataset_name, mode, samples, full_vol_dim, crop_size, sub_vol_path, normalization='max_min',
                       th_percent=0.1):
    """

    :param ls: list of modality paths, where the last path is the segmentation map
    :param dataset_name: which dataset is used
    :param mode: train/val
    :param samples: train/val samples to generate
    :param full_vol_dim: full image size
    :param crop_size: train volume size
    :param sub_vol_path: path for the particular patient
    :param th_percent: the % of the croped dim that corresponds to non-zero labels
    :param crop_type:
    :return:
    """
    total = len(ls[0])
    assert total != 0, "Problem reading data. Check the data paths."
    modalities = len(ls)
    list = []
    # print(modalities)
    # print(ls[2])

    print('Mode: ' + mode + ' Subvolume samples to generate: ', samples, ' Volumes: ', total)
    for i in range(samples):
        # print(i)
        random_index = np.random.randint(total)
        sample_paths = []
        tensor_images = []
        for j in range(modalities):
            sample_paths.append(ls[j][random_index])
        # print(sample_paths)
        while True:
            label_path = sample_paths[-1]
            crop = find_random_crop_dim(full_vol_dim, crop_size)
            full_segmentation_map = img_loader.load_medical_image(label_path, viz3d=True, type='label',
                                                                  crop_size=crop_size,
                                                                  crop=crop)

            full_segmentation_map = fix_seg_map(full_segmentation_map, dataset_name)
            # print(full_segmentation_map.shape)
            if find_non_zero_labels_mask(full_segmentation_map, th_percent, crop_size, crop):
                segmentation_map = img_loader.load_medical_image(label_path, type='label', crop_size=crop_size,
                                                                 crop=crop)
                segmentation_map = fix_seg_map(segmentation_map, dataset_name)
                for j in range(modalities - 1):
                    img_tensor = img_loader.load_medical_image(sample_paths[j], type="T1", normalization=normalization,
                                                               crop_size=crop_size, crop=crop)

                    tensor_images.append(img_tensor)

                break

        filename = sub_vol_path + 'id_' + str(random_index) + '_s_' + str(i) + '_modality_'
        list_saved_paths = []
        for j in range(modalities - 1):
            f_t1 = filename + str(j) + '.npy'
            list_saved_paths.append(f_t1)

            np.save(f_t1, tensor_images[j])

        f_seg = filename + 'seg.npy'

        np.save(f_seg, segmentation_map)
        list_saved_paths.append(f_seg)
        list.append(tuple(list_saved_paths))

    return list


def get_all_sub_volumes(*ls, dataset_name, mode, samples, full_vol_dim, crop_size, sub_vol_path,
                        normalization='max_min'):
    # TODO
    # 1.) gia ola tas subject fortwnwn image kai target
    # 2.) call generate_non_overlapping_volumes gia na kanw to image kai target sub_volumnes patches
    # 3.) apothikeuw tensors
    total = len(ls[0])
    assert total != 0, "Problem reading data. Check the data paths."
    modalities = len(ls)
    list = []

    for vol_id in range(total):

        tensor_images = []
        for modality_id in range(modalities - 1):
            img_tensor = img_loader.medical_image_transform(
                img_loader.load_medical_image(ls[modality_id][vol_id], type="T1"),
                normalization=normalization)

            img_tensor = generate_padded_subvolumes(img_tensor, kernel_dim=crop_size)

            tensor_images.append(img_tensor)
        segmentation_map = img_loader.medical_image_transform(
            img_loader.load_medical_image(ls[modalities - 1][vol_id], viz3d=True, type='label'))
        segmentation_map = generate_padded_subvolumes(segmentation_map, kernel_dim=crop_size)

        filename = sub_vol_path + 'id_' + str(vol_id) + '_s_' + str(modality_id) + '_modality_'

        list_saved_paths = []
        # print(len(tensor_images[0]))
        for k in range(len(tensor_images[0])):
            for j in range(modalities - 1):
                f_t1 = filename + str(j) + '_sample_{}'.format(str(k).zfill(8)) + '.npy'
                list_saved_paths.append(f_t1)
                # print(f_t1,tensor_images[j][k].shape)
                np.save(f_t1, tensor_images[j])

            f_seg = filename + 'seg_sample_{}'.format(str(k).zfill(8)) + '.npy'
            # print(f_seg)
            np.save(f_seg, segmentation_map)
            list_saved_paths.append(f_seg)
            list.append(tuple(list_saved_paths))

    # print(list)
    return list


def generate_padded_subvolumes(full_volume, kernel_dim=(32, 32, 32)):
    x = full_volume.detach()

    modalities, D, H, W = x.shape
    kc, kh, kw = kernel_dim
    dc, dh, dw = kernel_dim  # stride
    # Pad to multiples of kernel_dim
    a = ((roundup(W, kw) - W) // 2 + W % 2, (roundup(W, kw) - W) // 2,
         (roundup(H, kh) - H) // 2 + H % 2, (roundup(H, kh) - H) // 2,
         (roundup(D, kc) - D) // 2 + D % 2, (roundup(D, kc) - D) // 2)
    # print('padding ', a)
    x = F.pad(x, a)
    # print('padded shape ', x.shape)
    assert x.size(3) % kw == 0
    assert x.size(2) % kh == 0
    assert x.size(1) % kc == 0
    patches = x.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
    unfold_shape = list(patches.size())

    patches = patches.contiguous().view(-1, modalities, kc, kh, kw)

    return patches


def find_random_crop_dim(full_vol_dim, crop_size):
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


def find3Dlabel_boundaries(segmentation_map):
    target_indexs = np.where(segmentation_map > 0)
    maxs = np.max(np.array(target_indexs), axis=1)
    mins = np.min(np.array(target_indexs), axis=1)
    diff = maxs - mins
    labels_voxels = diff[0] * diff[1] * diff[2]
    return labels_voxels


def find_non_zero_labels_mask(segmentation_map, th_percent, crop_size, crop):
    d1, d2, d3 = segmentation_map.shape
    segmentation_map[segmentation_map > 0] = 1
    total_voxel_labels = segmentation_map.sum()

    cropped_segm_map = img_loader.crop_img(segmentation_map, crop_size, crop)
    crop_voxel_labels = cropped_segm_map.sum()

    label_percentage = crop_voxel_labels / total_voxel_labels
    # print(label_percentage,total_voxel_labels,crop_voxel_labels)
    if label_percentage >= th_percent:
        return True
    else:
        return False
