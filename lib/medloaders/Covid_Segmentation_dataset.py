import glob
import os

import numpy as np
from torch.utils.data import Dataset

import lib.utils as utils
from lib.medloaders import medical_image_process as img_loader


class COVID_Seg_Dataset(Dataset):
    """
    Code for reading the COVID Segmentation dataset

    Segmentation Task 1: Learning with limited annotations

        This task is based on the COVID-19-CT-Seg dataset with 20 cases.
        Three subtasks are to segment lung, infection or both of them.
        For each task, 5-fold cross-validation results should be reported.
        It should be noted that each fold only has 4 training cases, and remained 16 cases are used for testing.
        In other words, this is a few-shot or zero-shot segmentation task.
        Dataset split file and quantitative results of U-Net baseline are presented in Task1 folder.
    """

    def __init__(self, mode, sub_task='lung', split=0.2, fold=0, n_classes=3, samples=10, save=True,
                 dataset_path='/home/papastrat/PycharmProjects/MedicalZooPytorch/datasets',
                 crop_dim=(32, 32, 32)):
        print("COVID SEGMENTATION DATASET")
        self.CLASSES = n_classes
        self.fold = int(fold)
        self.crop_size = crop_dim
        self.full_vol_dim = (512, 512, 301)  # width, height,slice,
        self.mode = mode
        self.full_volume = None
        self.affine = None
        self.list = []
        self.samples = samples
        self.save = save
        subvol = '_vol_' + str(crop_dim[0]) + 'x' + str(crop_dim[1]) + 'x' + str(crop_dim[2])

        if self.save:
            self.sub_vol_path = dataset_path + '/covid_segmap_dataset/generated/' + mode + subvol + '/'
            utils.make_dirs(self.sub_vol_path)

        self.train_images, self.train_labels, self.val_labels, self.val_images = [], [], [], []
        list_images = sorted(
            glob.glob(os.path.join(dataset_path, 'covid_segmap_dataset/COVID-19-CT-Seg_20cases/*')))

        if (sub_task == 'lung'):
            list_labels = sorted(glob.glob(os.path.join(dataset_path, 'covid_segmap_dataset/Lung_Mask/*')))
        elif (sub_task == 'infection'):
            list_labels = sorted(glob.glob(os.path.join(dataset_path, 'covid_segmap_dataset/Infection_Mask/*')))
        else:
            list_labels = sorted(
                glob.glob(os.path.join(dataset_path, 'covid_segmap_dataset/Lung_and_Infection_Mask/*')))
        len_of_data = len(list_images)
        print(len_of_data)
        for i in range(len_of_data):
            if i >= (self.fold * int(split * len_of_data)) and i < (
                    (self.fold * int(split * len_of_data)) + int(split * len_of_data)):
                self.train_images.append(list_images[i])
                self.train_labels.append(list_labels[i])
            else:
                self.val_images.append(list_images[i])
                self.val_labels.append(list_labels[i])

        if (mode == 'train'):
            self.list_IDs = self.train_images
            self.list_labels = self.train_labels
            self.create_sub_volumes()
        elif (mode == 'val'):
            self.list_IDs = self.val_images
            self.list_labels = self.val_labels
            self.create_sub_volumes()
        print("{} SAMPLES =  {}".format(mode, len(self.list)))

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):


        if self.save:
            t1_path, seg_path = self.list[index]
            return np.load(t1_path), np.load(seg_path)
        # on-memory saved data
        else:
            return self.list[index]

    def create_sub_volumes(self):
        total = len(self.list_IDs)
        TH = 0  # threshold for non empty volumes
        print('Mode: ' + self.mode + ' Subvolume samples to generate: ', self.samples, ' Volumes: ', total)

        for i in range(self.samples):
            random_index = np.random.randint(total)
            path_T1 = self.list_IDs[random_index]

            while True:
                slices = np.random.randint(self.full_vol_dim[2] - self.crop_size[2])
                w_crop = np.random.randint(self.full_vol_dim[0] - self.crop_size[0])
                h_crop = np.random.randint(self.full_vol_dim[1] - self.crop_size[1])

                if self.list_labels is not None:
                    label_path = self.list_labels[random_index]
                    segmentation_map = img_loader.load_medical_image(label_path, crop_size=self.crop_size,
                                                                     crop=(w_crop, h_crop, slices), type='label')
                    if segmentation_map.sum() > TH:
                        img_t1_tensor = img_loader.load_medical_image(path_T1, crop_size=self.crop_size,
                                                                      crop=(w_crop, h_crop, slices), type="img")

                        # segmentation_map = self.fix_seg_map(segmentation_map)
                        break
                    else:
                        continue
                else:
                    segmentation_map = None
                    break

            if self.save:

                filename = self.sub_vol_path + 'id_' + str(random_index) + '_s_' + str(i) + '_'
                f_t1 = filename + 'img.npy'

                f_seg = filename + 'seg.npy'
                np.save(f_t1, img_t1_tensor)

                np.save(f_seg, segmentation_map)
                self.list.append(tuple((f_t1, f_seg)))
            else:
                self.list.append(tuple((img_t1_tensor, segmentation_map)))
#
# d = COVID_Seg_Dataset('train')
# x, y = d.__getitem__(0)
# print(y.max(),y.min())
