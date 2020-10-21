import glob
import os

import numpy as np
import torch

import medzoo.common.augment3D as augment3D
import medzoo.utils as utils
from medzoo.common.medloaders import medical_image_process as img_loader
from medzoo.common.medloaders.medical_loader_utils import get_viz_set, create_sub_volumes
from medzoo.datasets.dataset import MedzooDataset

class MRIDatasetISEG2019(MedzooDataset):
    """
    Code for reading the infant brain MRI dataset of ISEG 2017 challenge
    """

    def __init__(self, config, mode, dataset_path='./datasets'):

        super().__init__(config, mode, dataset_path)

        self.training_path = self.root_path + '/iseg_2019/iSeg-2019-Training/'
        self.testing_path = self.root_path + '/iseg_2019/iSeg-2019-Validation/'
        self.full_vol_dim = (144, 192, 256)  # slice, width, height

        self.save_name = self.root_path + '/iseg_2019/iseg2019-list-' + self.mode + '-samples-' + str(self.samples) + '.txt'

        self.list = []
        self.full_volume = None
        self.sub_vol_path = self.root_path + '/iseg_2019/generated/' + self.mode + self.subvol + '/'

        self.list_IDsT1 = None
        self.list_IDsT2 = None
        self.labels = None
        self.split_idx = int(self.split * 10)

        self.load_dataset()
        if self.augmentation:
            self.augment()
        else:
            self.transform = augment3D.Compose(
            [augment3D.ScaleIntensity(self.modality_keys),
             augment3D.AddChannelDim(self.modality_keys, apply_to_label=False),
             augment3D.DictToTensor(self.modality_keys),
             augment3D.DictToList()])
    def load(self):
        ## load pre-generated data
        self.list = utils.load_list(self.save_name)
        self.list_IDsT1 = sorted(glob.glob(os.path.join(self.training_path, '*T1.img')))
        self.affine = img_loader.load_affine_matrix(self.list_IDsT1[0])

    def preprocess(self):
        utils.make_dirs(self.sub_vol_path)

        self.list_IDsT1 = sorted(glob.glob(os.path.join(self.training_path, '*T1.img')))
        self.list_IDsT2 = sorted(glob.glob(os.path.join(self.training_path, '*T2.img')))
        self.labels = sorted(glob.glob(os.path.join(self.training_path, '*label.img')))
        self.affine = img_loader.load_affine_matrix(self.list_IDsT1[0])


    def preprocess_train(self):
        self.list_IDsT1 = self.list_IDsT1[:self.split_idx]
        self.list_IDsT2 = self.list_IDsT2[:self.split_idx]
        self.labels = self.labels[:self.split_idx]
        self.list = create_sub_volumes(self.list_IDsT1, self.list_IDsT2, self.labels, dataset_name="iseg2019",
                                       mode=self.mode, samples=self.samples, full_vol_dim=self.full_vol_dim,
                                       crop_size=self.crop_size,
                                       sub_vol_path=self.sub_vol_path, th_percent=self.threshold)

    def preprocess_val(self):
        list_IDsT1 = self.list_IDsT1[self.split_idx:]
        list_IDsT2 = self.list_IDsT2[:self.split_idx:]
        labels = self.labels[self.split_idx:]
        self.list = create_sub_volumes(list_IDsT1, list_IDsT2, labels, dataset_name="iseg2019",
                                       mode=self.mode, samples=self.samples, full_vol_dim=self.full_vol_dim,
                                       crop_size=self.crop_size,
                                       sub_vol_path=self.sub_vol_path, th_percent=self.threshold)

        self.full_volume = get_viz_set(list_IDsT1, list_IDsT2, labels, dataset_name="iseg2019")

    def preprocess_test(self):
        self.list_IDsT1 = sorted(glob.glob(os.path.join(self.testing_path, '*T1.img')))
        self.list_IDsT2 = sorted(glob.glob(os.path.join(self.testing_path, '*T2.img')))
        self.labels = None
        # todo inference here

    def augment(self):
        self.transform = augment3D.Compose(
            [augment3D.ScaleIntensity(self.modality_keys),
             augment3D.AddChannelDim(self.modality_keys, apply_to_label=False),
             augment3D.DictToTensor(self.modality_keys),
             augment3D.DictToList()])


    def save_list(self):
        utils.save_list(self.save_name, self.list)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):

        t1_path, t2_path, seg_path = self.list[index]

        data = {self.modality_keys[0]:np.load(t1_path),
                self.modality_keys[1]:np.load(t2_path),
                self.modality_keys[2]:np.load(seg_path)}
        input_tuple = self.transform(data)

        return input_tuple



