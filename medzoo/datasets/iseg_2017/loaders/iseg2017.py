import glob
import os

import numpy as np
import torch
import medzoo.common.augment3D as augment3D
import medzoo.utils as utils
from medzoo.common.medloaders import medical_image_process as img_loader
from medzoo.common.medloaders.medical_loader_utils import get_viz_set, create_sub_volumes
from medzoo.datasets.dataset import MedzooDataset

class MRIDatasetISEG2017(MedzooDataset):
    """
    Code for reading the infant brain MRI dataset of ISEG 2017 challenge
    """

    def __init__(self, config, mode, dataset_path='./datasets'):
        """

        Args:
            mode: 'train','val','test'
            dataset_path: root dataset folder
            crop_dim: subvolume tuple
            fold_id: 1 to 10 values
            samples: number of sub-volumes that you want to create
        """
        super().__init__( config, mode, root_path=dataset_path)

        self.training_path = self.root_path + '/iseg_2017/iSeg-2017-Training/'
        self.testing_path = self.root_path + '/iseg_2017/iSeg-2017-Testing/'

        self.samples = config[self.mode].total_samples

        self.save_name = self.root_path + '/iseg_2017/iSeg-2017-Training/iseg2017-list-' + mode + '-samples-' + str(
            self.samples) + '.txt'

        self.list = []
        self.full_volume = None
        self.sub_vol_path = self.root_path + '/iseg_2017/generated/' + self.mode + self.subvol + '/'

        self.list_IDsT1 = None
        self.list_IDsT2 = None
        self.labels = None
        self.split_idx = int(self.split * 10)

        self.transform = None

        self.load_dataset()


    def load(self):
        ## load pre-generated data
        self.list = utils.load_list(self.save_name)
        self.list_IDsT1 = sorted(glob.glob(os.path.join(self.training_path, '*T1.img')))
        self.affine = img_loader.load_affine_matrix(self.list_IDsT1[0])

    def augment(self):
        self.transform = augment3D.RandomChoice(
            transforms=[augment3D.GaussianNoise(mean=0, std=0.01), augment3D.RandomFlip(),
                        augment3D.ElasticTransform()], p=0.5)

    def preprocess(self):
        utils.make_dirs(self.sub_vol_path)
        self.list_IDsT1 = sorted(glob.glob(os.path.join(self.training_path, '*T1.img')))
        self.list_IDsT2 = sorted(glob.glob(os.path.join(self.training_path, '*T2.img')))
        self.labels = sorted(glob.glob(os.path.join(self.training_path, '*label.img')))
        self.affine = img_loader.load_affine_matrix(self.list_IDsT1[0])

    def preprocess_train(self):
        utils.make_dirs(self.sub_vol_path)

        self.list_IDsT1 = self.list_IDsT1[: self.split_idx]
        self.list_IDsT2 = self.list_IDsT2[: self.split_idx]
        self.labels = self.labels[: self.split_idx]

        self.list = create_sub_volumes(self.list_IDsT1, self.list_IDsT2, self.labels, dataset_name="iseg2017",
                                       mode=self.mode, samples=self.samples, full_vol_dim=self.full_vol_dim,
                                       crop_size=self.crop_size,
                                       sub_vol_path=self.sub_vol_path, th_percent=self.threshold,
                                       normalization=self.normalization)


    def preprocess_val(self):

        self.list_IDsT1 = self.list_IDsT1[self.split_idx :]
        self.list_IDsT2 = self.list_IDsT2[self.split_idx :]
        self.labels = self.labels[self.split_idx :]
        self.list = create_sub_volumes(self.list_IDsT1, self.list_IDsT2, self.labels, dataset_name="iseg2017",
                                       mode=self.mode, samples=self.samples, full_vol_dim=self.full_vol_dim,
                                       crop_size=self.crop_size,
                                       sub_vol_path=self.sub_vol_path, th_percent=self.threshold,
                                       normalization=self.normalization)

        self.full_volume = get_viz_set(self.list_IDsT1, self.list_IDsT2, self.labels, dataset_name="iseg2017")

    def preprocess_test(self):
        self.list_IDsT1 = sorted(glob.glob(os.path.join(self.testing_path, '*T1.img')))
        self.list_IDsT2 = sorted(glob.glob(os.path.join(self.testing_path, '*T2.img')))
        self.labels = None

    def preprocess_viz(self):
        self.list_IDsT1 = self.list_IDsT1[ self.split_idx:]
        self.list_IDsT2 = self.list_IDsT2[: self.split_idx:]
        self.labels = self.labels[ self.split_idx:]
        self.full_volume = get_viz_set(self.list_IDsT1, self.list_IDsT2, self.labels, dataset_name="iseg2017")

    def save_list(self):
        utils.save_list(self.save_name, self.list)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        t1_path, t2_path, seg_path = self.list[index]
        t1, t2, s = np.load(t1_path), np.load(t2_path), np.load(seg_path)

        if self.mode == 'train' and self.augmentation:

            [augmented_t1, augmented_t2], augmented_s = self.transform([t1, t2], s)

            return torch.FloatTensor(augmented_t1.copy()).unsqueeze(0), torch.FloatTensor(
                augmented_t2.copy()).unsqueeze(0), torch.FloatTensor(augmented_s.copy())

        return torch.FloatTensor(t1).unsqueeze(0), torch.FloatTensor(t2).unsqueeze(0), torch.FloatTensor(s)
