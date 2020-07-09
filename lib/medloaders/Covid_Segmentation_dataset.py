import glob
import os

import numpy as np
from torch.utils.data import Dataset

import lib.utils as utils
from lib.medloaders import medical_image_process as img_loader
from lib.medloaders.medical_loader_utils import create_sub_volumes


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

    def __init__(self, mode, sub_task='lung', split=0.2, fold=0, n_classes=3, samples=10, dataset_path='../datasets',
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
        subvol = '_vol_' + str(crop_dim[0]) + 'x' + str(crop_dim[1]) + 'x' + str(crop_dim[2])

        self.sub_vol_path = dataset_path + '/covid_segmap_dataset/generated/' + mode + subvol + '/'
        utils.make_dirs(self.sub_vol_path)

        self.train_images, self.train_labels, self.val_labels, self.val_images = [], [], [], []
        list_images = sorted(
            glob.glob(os.path.join(dataset_path, 'covid_segmap_dataset/COVID-19-CT-Seg_20cases/*')))

        if sub_task == 'lung':
            list_labels = sorted(glob.glob(os.path.join(dataset_path, 'covid_segmap_dataset/Lung_Mask/*')))
        elif sub_task == 'infection':
            list_labels = sorted(glob.glob(os.path.join(dataset_path, 'covid_segmap_dataset/Infection_Mask/*')))
        else:
            list_labels = sorted(
                glob.glob(os.path.join(dataset_path, 'covid_segmap_dataset/Lung_and_Infection_Mask/*')))
        len_of_data = len(list_images)

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

        elif (mode == 'val'):
            self.list_IDs = self.val_images
            self.list_labels = self.val_labels

        self.list = create_sub_volumes(self.list_IDs, self.list_labels, dataset_name='covid19seg', mode=mode,
                                       samples=samples, full_vol_dim=self.full_vol_dim, crop_size=self.crop_size,
                                       sub_vol_path=self.sub_vol_path)
        print("{} SAMPLES =  {}".format(mode, len(self.list)))

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        t1_path, seg_path = self.list[index]
        return np.load(t1_path), np.load(seg_path)
