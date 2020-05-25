import numpy as np

from lib.medloaders import medical_image_process as img_loader
from lib.visual3D_temp import *


def get_viz_set(*ls,dataset_name, test_subject=0, save=False, sub_vol_path=None):
    """
    Returns total 3d input volumes (t1 and t2 or more) and segmentation maps
    3d total vol shape : torch.Size([1, 144, 192, 256])
    """
    modalities = len(ls)
    total_volumes = []

    for i in range(modalities):
        path_img = ls[i][test_subject]

        img_tensor = img_loader.load_medical_image(path_img, viz3d=True)
        if i == modalities-1:

            img_tensor = fix_seg_map(img_tensor,dataset=dataset_name)

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

    elif dataset == "brats2018" or dataset == "brats2019":
        segmentation_map[segmentation_map == 3] = 5
        segmentation_map[segmentation_map == 4] = 3
        segmentation_map[segmentation_map >= 4] = 4
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


def create_sub_volumes(*ls, dataset_name, mode, samples, full_vol_dim, crop_size, sub_vol_path, threshold=10):
    total = len(ls[0])
    assert total != 0, "Problem reading data. Check the data paths."
    modalities = len(ls)
    list = []
    print('Mode: ' + mode + ' Subvolume samples to generate: ', samples, ' Volumes: ', total)
    for i in range(samples):
        random_index = np.random.randint(total)
        sample_paths = []
        tensor_images = []
        for j in range(modalities):
            sample_paths.append(ls[j][random_index])

        while True:
            crop = find_random_crop_dim(full_vol_dim, crop_size)

            label_path = sample_paths[-1]
            segmentation_map = img_loader.load_medical_image(label_path, crop_size=crop_size,
                                                             crop=crop, type='label')

            segmentation_map = fix_seg_map(segmentation_map, dataset_name)
            if segmentation_map.sum() > threshold:
                for j in range(modalities - 1):
                    img_tensor = img_loader.load_medical_image(sample_paths[j], crop_size=crop_size,
                                                               crop=crop, type="T1")
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
