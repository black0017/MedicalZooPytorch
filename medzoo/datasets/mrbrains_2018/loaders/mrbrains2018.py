import glob
import os

import numpy as np

import medzoo.utils as utils
from medzoo.common.medloaders import medical_image_process as img_loader
from medzoo.common.medloaders.medical_loader_utils import create_sub_volumes
from medzoo.common.medloaders.medical_loader_utils import get_viz_set
from medzoo.datasets.dataset import MedzooDataset


class MRIDatasetMRBRAINS2018(MedzooDataset):
    """

    """

    def __init__(self, config, mode, dataset_path='../datasets'):
        super().__init__(config, mode, dataset_path)

        self.dataset_name = "mrbrains" + str(self.classes)
        self.training_path = self.root_path + '/mrbrains_2018/training'
        self.dirs = os.listdir(self.training_path)
        self.list = []
        self.threshold = 0.1
        self.list_flair = []
        self.list_ir = []
        self.list_reg_ir = []
        self.list_reg_t1 = []
        self.labels = []
        self.full_volume = None

        self.split_idx = 0

        self.save_name = self.root_path + '/mrbrains_2018/training/mrbrains_2018-classes-' + str(
            self.classes) + '-list-' + mode + '-samples-' + str(
            self.samples) + '.txt'

        self.sub_vol_path = self.root_path + '/mrbrains_2018/generated/' + mode + self.subvol + '/'

        self.load_dataset()


    def load(self):
        ## load pre-generated data
        self.list = utils.load_list(self.save_name)
        utils.make_dirs(self.sub_vol_path)

    def preprocess(self):
        list_reg_t1 = sorted(glob.glob(os.path.join(self.training_path, '*/pr*/*g_T1.nii.gz')))
        list_reg_ir = sorted(glob.glob(os.path.join(self.training_path, '*/pr*/*g_IR.nii.gz')))
        list_flair = sorted(glob.glob(os.path.join(self.training_path, '*/pr*/*AIR.nii.gz')))
        labels = sorted(glob.glob(os.path.join(self.training_path, '*/*egm.nii.gz')))

        self.affine = img_loader.load_affine_matrix(list_reg_t1[0])

    def preprocess_val(self):
        labels = [self.labels[self.split_idx]]
        list_reg_t1 = [self.list_reg_t1[self.split_idx]]
        list_reg_ir = [self.list_reg_ir[self.split_idx]]
        list_flair = [self.list_flair[self.split_idx]]
        self.full_volume = get_viz_set(list_reg_t1, list_reg_ir, list_flair, labels, dataset_name=self.dataset_name)

        self.list = create_sub_volumes(self.list_reg_t1, self.list_reg_ir, self.list_flair, self.labels,
                                       dataset_name=self.dataset_name, mode=self.mode,
                                       samples=self.samples, full_vol_dim=self.full_vol_dim,
                                       crop_size=self.crop_size, sub_vol_path=self.sub_vol_path,
                                       th_percent=self.threshold)

    def preprocess_train(self):
        self.labels.pop(self.split_idx)
        self.list_reg_t1.pop(self.split_idx)
        self.list_reg_ir.pop(self.split_idx)
        self.list_flair.pop(self.split_idx)

        self.list = create_sub_volumes(self.list_reg_t1, self.list_reg_ir, self.list_flair, self.labels,
                                       dataset_name=self.dataset_name, mode=self.mode,
                                       samples=self.samples, full_vol_dim=self.full_vol_dim,
                                       crop_size=self.crop_size, sub_vol_path=self.sub_vol_path,
                                       th_percent=self.threshold)

    def save_list(self):
        utils.save_list(self.save_name, self.list)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        t1_path, ir_path, flair_path, seg_path = self.list[index]
        return np.load(t1_path), np.load(ir_path), np.load(flair_path), np.load(seg_path)
