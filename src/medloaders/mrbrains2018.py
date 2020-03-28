import os
from torch.utils.data import Dataset
import glob
import numpy as np

from src.medloaders import img_loader
import src.utils as utils


class MRIDatasetMRBRAINS2018(Dataset):
    def __init__(self, mode, dataset_path='../datasets', dim=(32, 32, 32), fold_id=1, classes=4, samples=1000,
                 save=True):
        self.mode = mode
        self.root = dataset_path
        self.training_path = self.root + '/mrbrains_2018/training'
        self.dirs = os.listdir(self.training_path)
        self.CLASSES = classes
        self.samples = samples
        self.save = save
        self.list = []
        self.full_vol_size = (240, 240, 48)
        self.crop_dim = dim
        self.fold = str(fold_id)
        self.list_flair = []
        self.list_ir = []
        self.list_reg_ir = []
        self.list_reg_t1 = []
        self.labels = []
        self.full_volume = None

        if self.save:
            subvol = '_vol_' + str(dim[0]) + 'x' + str(dim[1]) + 'x' + str(dim[2])
            self.sub_vol_path = self.root + '/mrbrains_2018/generated/' + mode + subvol + '/'
            utils.make_dirs(self.sub_vol_path)

        for counter, i in enumerate(self.dirs):
            path_seg = self.training_path + "/" + i
            path_img = self.training_path + "/" + i + "/pre/"

            if str(counter) == fold_id and mode == 'val':
                self.labels.append(path_seg + "/segm.nii.gz")
                self.list_flair.append(path_img + "FLAIR.nii.gz")
                self.list_reg_ir.append(path_img + "reg_IR.nii.gz")
                self.list_reg_t1.append(path_img + "reg_T1.nii.gz")
            elif str(counter) != fold_id and mode == "train":
                self.labels.append(path_seg + "/segm.nii.gz")
                self.list_flair.append(path_img + "FLAIR.nii.gz")
                self.list_reg_ir.append(path_img + "reg_IR.nii.gz")
                self.list_reg_t1.append(path_img + "reg_T1.nii.gz")

        self.affine = img_loader.load_affine_matrix(self.list_reg_t1[0])
        self.get_samples()

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        # offline data
        if self.save:
            t1_path, ir_path, flair_path, seg_path = self.list[index]
            return np.load(t1_path), np.load(ir_path), np.load(flair_path), np.load(seg_path)
        # on-memory saved data
        else:
            return self.list[index]

    def fix_seg_map(self, segmentation_map, classes=4):
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
        TH = 10
        total = len(self.labels)
        print('Mode: ' + self.mode + ' Subvolume samples to generate: ', self.samples, ' Volumes: ', total)
        for i in range(self.samples):
            random_index = np.random.randint(total)

            path_flair = self.list_flair[random_index]
            path_reg_ir = self.list_reg_ir[random_index]
            path_reg_t1 = self.list_reg_t1[random_index]

            while True:
                w_crop = np.random.randint(self.full_vol_size[0] - self.crop_dim[0])
                h_crop = np.random.randint(self.full_vol_size[1] - self.crop_dim[1])
                slices = np.random.randint(self.full_vol_size[2] - self.crop_dim[2])
                crop = (w_crop, h_crop, slices)

                if self.labels is not None:
                    label_path = self.labels[random_index]
                    segmentation_map = img_loader.load_medical_image(label_path, crop_size=self.crop_dim,
                                                                     crop=crop, type='label')
                    segmentation_map = self.fix_seg_map(segmentation_map)

                    if segmentation_map.sum() > TH:
                        img_t1_tensor = img_loader.load_medical_image(path_reg_t1, crop_size=self.crop_dim,
                                                                      crop=crop,
                                                                      type="T1")
                        img_ir_tensor = img_loader.load_medical_image(path_reg_ir, crop_size=self.crop_dim,
                                                                      crop=crop,
                                                                      type="reg-IR")
                        img_flair_tensor = img_loader.load_medical_image(path_flair, crop_size=self.crop_dim,
                                                                         crop=crop,
                                                                         type="FLAIR")
                        break
                    else:
                        continue
                else:
                    segmentation_map = None
                    break
            if self.save:
                filename = self.sub_vol_path + 'id_' + str(random_index) + '_s_' + str(i) + '_'
                f_t1 = filename + 'T1.npy'
                f_ir = filename + 'IR.npy'
                f_flair = filename + 'FLAIR.npy'
                f_seg = filename + 'seg.npy'

                np.save(f_t1, img_t1_tensor)
                np.save(f_ir, img_ir_tensor)
                np.save(f_flair, img_flair_tensor)
                np.save(f_seg, segmentation_map)

                self.list.append(tuple((f_t1, f_ir, f_flair, f_seg)))
            else:
                self.list.append(tuple((img_t1_tensor, img_ir_tensor, img_flair_tensor, segmentation_map)))

    def get_viz_set(self):
        """
        Returns total 3d input volumes(t1 and t2) and segmentation maps
        3d total vol shape : torch.Size([1, 144, 192, 256])
        """
        path_t1 = self.list_reg_t1[self.fold]
        path_ir = self.list_ir[self.fold]
        path_flair = self.list_flair[self.fold]
        label_path = self.labels[self.fold]
        segmentation_map = img_loader.load_medical_image(label_path, type="label", viz3d=True)
        img_t1_tensor = img_loader.load_medical_image(path_t1, type="T1", viz3d=True)
        img_ir_tensor = img_loader.load_medical_image(path_ir, type="T2", viz3d=True)
        img_flair_tensor = img_loader.load_medical_image(path_flair, type="FLAIR", viz3d=True)
        segmentation_map = self.fix_seg_map(segmentation_map)
        self.full_volume = tuple((img_t1_tensor, img_ir_tensor, img_flair_tensor, segmentation_map))
        print("Full validation volume has been generated")
