import os
import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import nibabel as nib

import src.utils as utils

#TODO

class MRIDatasetMRBRAINS2018(Dataset):
    def __init__(self, mode, dataset_path='./data', dim=(32, 32, 32), fold_id=1, classes=4, samples=1000):
        self.debug = False
        self.mode = mode
        self.root = dataset_path
        self.training_path = self.root + '/mrbrains'
        self.dirs = os.listdir(self.training_path)
        self.CLASSES = classes
        self.samples = samples
        self.list = []

        self.height = 240
        self.width = 240
        self.slices = 48

        self.crop_height = dim[0]
        self.crop_width = dim[1]
        self.crop_slices = dim[2]
        self.fold = str(fold_id)

        self.list_flair = []
        self.list_ir = []
        self.list_reg_ir = []
        self.list_reg_t1 = []
        self.list_t1 = []
        self.labels = []

        list_flair_tr = []
        list_ir_tr = []
        list_reg_ir_tr = []
        list_reg_t1_tr = []
        list_t1_tr = []
        labels_tr = []

        list_flair_val = []
        list_ir_val = []
        list_reg_ir_val = []
        list_reg_t1_val = []
        list_t1_val = []
        labels_val = []

        for counter, i in enumerate(self.dirs):
            path_seg = self.training_path + "/" + i
            path_img = self.training_path + "/" + i + "/pre/"

            if (str(counter) == fold_id):
                labels_val.append(path_seg + "/segm.nii.gz")
                list_flair_val.append(path_img + "FLAIR.nii.gz")
                list_ir_val.append(path_img + "IR.nii.gz")
                list_reg_ir_val.append(path_img + "reg_IR.nii.gz")
                list_reg_t1_val.append(path_img + "reg_T1.nii.gz")
                list_t1_val.append(path_img + "T1.nii.gz")
            else:
                labels_tr.append(path_seg + "/segm.nii.gz")
                list_flair_tr.append(path_img + "FLAIR.nii.gz")
                list_ir_tr.append(path_img + "IR.nii.gz")
                list_reg_ir_tr.append(path_img + "reg_IR.nii.gz")
                list_reg_t1_tr.append(path_img + "reg_T1.nii.gz")
                list_t1_tr.append(path_img + "T1.nii.gz")

        if (self.mode == 'train'):
            self.list_flair = list_flair_tr
            self.list_ir = list_ir_tr
            self.list_reg_ir = list_reg_ir_tr
            self.list_reg_t1 = list_reg_t1_tr
            self.list_t1 = list_t1_tr
            self.labels = labels_tr
        elif (self.mode == 'val'):
            self.list_flair = list_flair_val
            self.list_ir = list_ir_val
            self.list_reg_ir = list_reg_ir_val
            self.list_reg_t1 = list_reg_t1_val
            self.list_t1 = list_t1_val
            self.labels = labels_val

        self.get_samples()

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        # print('index = ', index)
        # print('self.samp12345les = ', self.samples)
        if index <= self.samples:
            tuple_in = self.list[index]
            return tuple_in
            # img_reg_t1_tensor, img_reg_ir_tensor,img_ir_tensor,img_flair_tensor, segmentation_map = tuple_in
            # return img_t1_tensor, img_t2_tensor, segmentation_map
        else:
            print("E.K.M.EK")
            return None

    def load_medical_image(self, path, crop, type=None, normalization="mean"):
        slices, w_crop, h_crop = crop
        img = nib.load(path)
        img_np = np.squeeze(img.get_fdata()).transpose(2, 0, 1)  # slices are first!!!
        img_np = img_np[slices:slices + self.crop_slices, w_crop:w_crop + self.crop_width,
                 h_crop:h_crop + self.crop_height]
        img_tensor = torch.from_numpy(img_np).float()
        if (type != 'label'):
            if normalization == "mean":
                mask = img_tensor.ne(0.0)
                desired = img_tensor[mask]
                mean_val, std_val = desired.mean(), desired.std()
                img_tensor = (img_tensor - mean_val) / std_val
            else:
                img_tensor = img_tensor / 255.0
            return img_tensor.unsqueeze(0)
        return img_tensor

    def fix_seg_map(self, segmentation_map):
        GM = 1
        WM = 2
        CSF = 3
        OTHER = 7
        segmentation_map[segmentation_map == 1] = GM
        segmentation_map[segmentation_map == 2] = GM
        segmentation_map[segmentation_map == 3] = WM
        segmentation_map[segmentation_map == 4] = WM
        segmentation_map[segmentation_map == 5] = CSF
        segmentation_map[segmentation_map == 6] = CSF
        segmentation_map[segmentation_map >= 7] = OTHER
        return segmentation_map

    def get_samples(self):
        MIN_SEG_VOXELS = 5
        total = len(self.labels)
        print('total ' + self.mode + ' data to generate samples ', total, 'samples=', self.samples)
        for i in range(self.samples):
            # print('i=',i)
            random_index = np.random.randint(total)

            path_flair = self.list_flair[random_index]
            path_ir = self.list_ir[random_index]
            path_reg_ir = self.list_reg_ir[random_index]
            path_reg_t1 = self.list_reg_t1[random_index]

            while True:
                h_crop = np.random.randint(self.height - self.crop_height)
                w_crop = np.random.randint(self.width - self.crop_width)
                slices = np.random.randint(self.slices - self.crop_slices)

                if self.labels is not None:
                    label_path = self.labels[random_index]
                    segmentation_map = self.load_medical_image(label_path, crop=(slices, w_crop, h_crop), type='label')
                    segmentation_map = self.fix_seg_map(segmentation_map)

                    if segmentation_map.sum() > MIN_SEG_VOXELS:
                        img_reg_t1_tensor = self.load_medical_image(path_reg_t1, crop=(slices, w_crop, h_crop),
                                                                    type="T1")
                        img_reg_ir_tensor = self.load_medical_image(path_reg_ir, crop=(slices, w_crop, h_crop),
                                                                    type="reg-IR")
                        img_ir_tensor = self.load_medical_image(path_ir, crop=(slices, w_crop, h_crop), type="IR")
                        img_flair_tensor = self.load_medical_image(path_flair, crop=(slices, w_crop, h_crop),
                                                                   type="FLAIR")
                        break
                    else:
                        continue
                else:
                    segmentation_map = None

                    break
            self.list.append(tuple(
                (img_reg_t1_tensor, img_reg_ir_tensor, img_ir_tensor, img_flair_tensor, segmentation_map)))
            # print('appended')

