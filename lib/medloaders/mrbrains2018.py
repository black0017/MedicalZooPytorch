import glob
import os
import torch
import numpy as np
from torch.utils.data import Dataset

import lib.utils as utils
from lib.medloaders import medical_image_process as img_loader
from lib.medloaders.medical_loader_utils import create_sub_volumes
from lib.medloaders.medical_loader_utils import get_viz_set
import lib.augment3D as augment3D

class MRIDatasetMRBRAINS2018(Dataset):
    def __init__(self, args, mode, dataset_path='../datasets', classes=4, dim=(32, 32, 32), split_id=0, samples=1000,
                 load=False):
        self.mode = mode
        self.root = dataset_path
        self.classes = classes
        dataset_name = "mrbrains" + str(classes)
        self.training_path = self.root + '/mrbrains_2018/training'
        self.dirs = os.listdir(self.training_path)
        self.samples = samples
        self.list = []
        self.full_vol_size = (240, 240, 48)
        self.threshold = args.threshold
        self.normalization = args.normalization
        self.augmentation = args.augmentation
        self.crop_dim = dim
        self.list_flair = []
        self.list_ir = []
        self.list_reg_ir = []
        self.list_reg_t1 = []
        self.labels = []
        self.full_volume = None

        list_reg_t1 = sorted(glob.glob(os.path.join(self.training_path, '*/pr*/*g_T1.nii.gz')))
        list_reg_ir = sorted(glob.glob(os.path.join(self.training_path, '*/pr*/*g_IR.nii.gz')))
        list_flair = sorted(glob.glob(os.path.join(self.training_path, '*/pr*/*AIR.nii.gz')))
        labels = sorted(glob.glob(os.path.join(self.training_path, '*/*egm.nii.gz')))
        list_reg_t1, list_reg_ir, list_flair, labels = utils.shuffle_lists(list_reg_t1, list_reg_ir, list_flair, labels)

        self.affine = img_loader.load_affine_matrix(list_reg_t1[0])
        self.full_volume = get_viz_set(list_reg_t1, list_reg_ir, list_flair, labels, dataset_name=dataset_name)
        if self.augmentation:
            self.transform = augment3D.RandomChoice(
                transforms=[augment3D.GaussianNoise(mean=0, std=0.01), augment3D.RandomFlip(),
                            augment3D.ElasticTransform()], p=0.5)

        self.save_name = self.root + '/mrbrains_2018/training/mrbrains_2018-classes-' + str(
            classes) + '-list-' + mode + '-samples-' + str(
            samples) + str(dim[0]) + 'x' + str(dim[1]) + 'x' + str(dim[2]) + '.txt'

        if load:
            ## load pre-generated data
            self.list = utils.load_list(self.save_name)
            return

        subvol = '_vol_' + str(dim[0]) + 'x' + str(dim[1]) + 'x' + str(dim[2])
        self.sub_vol_path = self.root + '/mrbrains_2018/generated/' + mode + subvol + '/'
        utils.make_dirs(self.sub_vol_path)

        val_split_id = 1
        test_split_id = 0
        if mode == 'val':
            labels = [labels[val_split_id]]
            list_reg_t1 = [list_reg_t1[val_split_id]]
            list_reg_ir = [list_reg_ir[val_split_id]]
            list_flair = [list_flair[val_split_id]]
            self.full_volume = get_viz_set(list_reg_t1, list_reg_ir, list_flair, labels, dataset_name=dataset_name)

        elif mode == 'test':
            labels = [labels[test_split_id]]
            list_reg_t1 = [list_reg_t1[test_split_id]]
            list_reg_ir = [list_reg_ir[test_split_id]]
            list_flair = [list_flair[test_split_id]]
            self.full_volume = get_viz_set(list_reg_t1, list_reg_ir, list_flair, labels, dataset_name=dataset_name)
        else:
            labels = labels[2:]
            list_reg_t1 = list_reg_t1[2:]
            list_reg_ir = list_reg_ir[2:]
            list_flair = list_flair[2:]

        self.list = create_sub_volumes(list_reg_t1, list_reg_ir, list_flair, labels,
                                       dataset_name=dataset_name, mode=mode,
                                       samples=samples, full_vol_dim=self.full_vol_size,
                                       crop_size=self.crop_dim, sub_vol_path=self.sub_vol_path,
                                       th_percent=self.threshold)

        utils.save_list(self.save_name, self.list)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        t1_path, ir_path, flair_path, seg_path = self.list[index]

        img_t1, img_ir, img_flair, img_seg = np.load(t1_path), np.load(ir_path), np.load(flair_path), np.load(seg_path)
        if self.mode == 'train' and self.augmentation:
            [img_t1, img_ir, img_flair], img_seg = self.transform([img_t1, img_ir, img_flair],
                                                                  img_seg)
        return torch.tensor(img_t1.copy()).unsqueeze(0), torch.tensor(img_ir.copy()).unsqueeze(
            0), torch.tensor(img_flair.copy()).unsqueeze(
            0), torch.tensor(img_seg.copy())
