import glob
import os

import numpy as np

import medzoo.utils as utils
from medzoo.common.medloaders.medical_loader_utils import create_sub_volumes
from medzoo.datasets.dataset import MedzooDataset


class COVID_Seg_Dataset(MedzooDataset):
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

    def __init__(self, config, mode,  dataset_path='./datasets'):
        super().__init__(config, mode, dataset_path)

        print("COVID SEGMENTATION DATASET")

        self.full_volume = None
        self.list = []

        self.sub_vol_path = dataset_path + '/covid_segmap_dataset/generated/' + mode + self.subvol + '/'

        self.split_idx= 0.2
        self.sub_task = 'lung'

        self.train_images, self.train_labels, self.val_labels, self.val_images = [], [], [], []

        self.list_IDs = None
        self.list_labels = None


    def load(self):
        utils.make_dirs(self.sub_vol_path)
        self.list = create_sub_volumes(self.list_IDs, self.list_labels, dataset_name='covid19seg', mode=self.mode,
                                       samples=self.samples, full_vol_dim=self.full_vol_dim, crop_size=self.crop_size,
                                       sub_vol_path=self.sub_vol_path)
        print("{} SAMPLES =  {}".format(self.mode, len(self.list)))

    def preprocess(self):
        list_images = sorted(
            glob.glob(os.path.join(self.root_path, 'covid_segmap_dataset/COVID-19-CT-Seg_20cases/*')))

        if self.sub_task == 'lung':
            list_labels = sorted(glob.glob(os.path.join(self.root_path, 'covid_segmap_dataset/Lung_Mask/*')))
        elif self.sub_task == 'infection':
            list_labels = sorted(glob.glob(os.path.join(self.root_path, 'covid_segmap_dataset/Infection_Mask/*')))
        else:
            list_labels = sorted(
                glob.glob(os.path.join(self.root_path, 'covid_segmap_dataset/Lung_and_Infection_Mask/*')))
        len_of_data = len(list_images)

        for i in range(len_of_data):
            if i >= (self.fold * int( self.split_idx * len_of_data)) and i < (
                    (self.fold * int( self.split_idx * len_of_data)) + int( self.split_idx * len_of_data)):
                self.train_images.append(list_images[i])
                self.train_labels.append(list_labels[i])
            else:
                self.val_images.append(list_images[i])
                self.val_labels.append(list_labels[i])

    def preprocess_train(self):
        self.list_IDs = self.train_images
        self.list_labels = self.train_labels

    def preprocess_val(self):
        self.list_IDs = self.val_images
        self.list_labels = self.val_labels

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        t1_path, seg_path = self.list[index]
        return np.load(t1_path), np.load(seg_path)
