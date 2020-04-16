import os
from torch.utils.data import Dataset
import glob
import numpy as np

#from lib.medloaders import img_loader
from lib.medloaders import medical_image_process as img_loader
import lib.utils as utils


class MICCAIBraTS2018(Dataset):
    """
    Code for reading the infant brain MICCAIBraTS2018 challenge
    """

    def __init__(self, mode, dataset_path='./datasets', classes=5, crop_dim=(32, 32, 32), split_idx=10, samples=10,
                 save=True):
        """
        :param mode: 'train','val','test'
        :param dataset_path: root dataset folder
        :param crop_dim: subvolume tuple
        :param split_idx: 1 to 10 values
        :param samples: number of sub-volumes that you want to create
        """
        self.mode = mode
        self.root = str(dataset_path)
        self.training_path = self.root + '/MICCAI_BraTS_2018_Data_Training/'
        self.testing_path = self.root + ' '
        self.save = save
        self.CLASSES = 4
        self.full_vol_dim = (240, 240, 155)  # slice, width, height
        self.crop_size = crop_dim
        self.list = []
        self.samples = samples
        self.full_volume = None
        self.classes = classes

        subvol = '_vol_' + str(crop_dim[0]) + 'x' + str(crop_dim[1]) + 'x' + str(crop_dim[2])

        if self.save:
            self.sub_vol_path = self.root + '/MICCAI_BraTS_2018_Data_Training/generated/' + mode + subvol + '/'
            utils.make_dirs(self.sub_vol_path)
        # print(self.training_path)
        list_IDsT1 = sorted(glob.glob(os.path.join(self.training_path, '*GG/*/*t1.nii.gz')))
        list_IDsT1ce = sorted(glob.glob(os.path.join(self.training_path, '*GG/*/*t1ce.nii.gz')))
        list_IDsT2 = sorted(glob.glob(os.path.join(self.training_path, '*GG/*/*t2.nii.gz')))
        list_IDsFlair = sorted(glob.glob(os.path.join(self.training_path, '*GG/*/*_flair.nii.gz')))
        labels = sorted(glob.glob(os.path.join(self.training_path, '*GG/*/*_seg.nii.gz')))

        # TODO shuffle lists

        self.affine = img_loader.load_affine_matrix(list_IDsT1[0])

        if self.mode == 'train':
            self.list_IDsT1 = list_IDsT1[:split_idx]
            self.list_IDsT1ce = list_IDsT1ce[:split_idx]
            self.list_IDsT2 = list_IDsT2[:split_idx]
            self.list_IDsFlair = list_IDsFlair[:split_idx]
            self.labels = labels[:split_idx]

            # Generates datasets with non-empty sub-volumes!
            self.create_sub_volumes()

        elif self.mode == 'val':
            self.list_IDsT1 = list_IDsT1[split_idx:]
            self.list_IDsT1ce = list_IDsT1ce[split_idx:]
            self.list_IDsT2 = list_IDsT2[split_idx:]
            self.list_IDsFlair = list_IDsFlair[split_idx:]
            self.labels = labels[split_idx:]

            self.create_sub_volumes()

            # TODO
            # self.get_viz_set()

        elif self.mode == 'test':
            self.list_IDsT1 = sorted(glob.glob(os.path.join(self.testing_path, '*GG/*/*t1.nii.gz')))
            self.list_IDsT1ce = sorted(glob.glob(os.path.join(self.testing_path, '*GG/*/*t1ce.nii.gz')))
            self.list_IDsT2 = sorted(glob.glob(os.path.join(self.testing_path, '*GG/*/*t2.nii.gz')))
            self.list_IDsFlair = sorted(glob.glob(os.path.join(self.testing_path, '*GG/*/*_flair.nii.gz')))
            self.labels = None

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        # offline data
        if self.save:
            f_t1, f_t1ce, f_t2, f_flair, f_seg = self.list[index]
            return np.load(f_t1), np.load(f_t1ce), np.load(f_t2), np.load(f_flair), np.load(f_seg)
        # on-memory saved data
        else:
            return self.list[index]

    def create_sub_volumes(self):
        total = len(self.list_IDsT1)
        TH = 10  # threshold for non empty volumes
        print('Mode: ' + self.mode + ' Subvolume samples to generate: ', self.samples, ' Volumes: ', total)

        for i in range(self.samples):
            random_index = np.random.randint(total)
            path_T1 = self.list_IDsT1[random_index]
            path_T1ce = self.list_IDsT1ce[random_index]
            path_T2 = self.list_IDsT2[random_index]
            path_flair = self.list_IDsFlair[random_index]

            while True:
                slices = np.random.randint(self.full_vol_dim[0] - self.crop_size[0])
                w_crop = np.random.randint(self.full_vol_dim[1] - self.crop_size[1])
                h_crop = np.random.randint(self.full_vol_dim[2] - self.crop_size[2])
                crop = (slices, w_crop, h_crop)

                if self.labels is not None:
                    label_path = self.labels[random_index]
                    segmentation_map = img_loader.load_medical_image(label_path, crop_size=self.crop_size,
                                                                     crop=crop, type='label')
                    if segmentation_map.sum() > TH:
                        img_t1_tensor = img_loader.load_medical_image(path_T1, crop_size=self.crop_size,
                                                                      crop=crop, type="T1")
                        img_t1ce_tensor = img_loader.load_medical_image(path_T1ce, crop_size=self.crop_size,
                                                                        crop=crop, type="T1ce")
                        img_t2_tensor = img_loader.load_medical_image(path_T2, crop_size=self.crop_size,
                                                                      crop=crop, type="T2")
                        img_flair_tensor = img_loader.load_medical_image(path_flair, crop_size=self.crop_size,
                                                                         crop=crop, type="flair")

                        break
                    else:
                        continue
                else:
                    segmentation_map = None
                    break

            if self.save:

                filename = self.sub_vol_path + 'id_' + str(random_index) + '_s_' + str(i) + '_'
                f_t1 = filename + 'T1.npy'
                f_t1ce = filename + 'T1CE.npy'
                f_t2 = filename + 'T2.npy'
                f_flair = filename + 'FLAIR.npy'
                f_seg = filename + 'seg.npy'

                np.save(f_t1, img_t1_tensor)
                np.save(f_t1ce, img_t1ce_tensor)
                np.save(f_t2, img_t2_tensor)
                np.save(f_flair, img_flair_tensor)
                np.save(f_seg, segmentation_map)
                self.list.append(tuple((f_t1, f_t1ce, f_t2, f_flair, f_seg)))
            else:
                self.list.append(
                    tuple((img_t1_tensor, img_t1ce_tensor, img_t2_tensor, img_flair_tensor, segmentation_map)))

    def get_viz_set(self, test_subject=0):
        """
        Returns total 3d input volumes (t1 and t2 or more) and segmentation maps
        3d total vol shape : torch.Size([1, 144, 192, 256])
        """
        TEST_SUBJECT = test_subject
        path_T1 = self.list_IDsT1[TEST_SUBJECT]
        path_T1ce = self.list_IDsT1ce[TEST_SUBJECT]
        path_T2 = self.list_IDsT2[TEST_SUBJECT]
        path_flair = self.list_IDsFlair[TEST_SUBJECT]
        label_path = self.labels[TEST_SUBJECT]

        segmentation_map = img_loader.load_medical_image(label_path, viz3d=True)
        img_t1_tensor = img_loader.load_medical_image(path_T1, type="T1", viz3d=True)
        img_t1ce_tensor = img_loader.load_medical_image(path_T1ce, type="T1ce", viz3d=True)
        img_t2_tensor = img_loader.load_medical_image(path_T2, type="T2", viz3d=True)
        img_flair_tensor = img_loader.load_medical_image(path_flair, type="FLAIR", viz3d=True)

        ### TO DO SAVE FULL VOLUME AS numpy
        if self.save:
            self.full_volume = []

            segmentation_map = segmentation_map
            img_t1_tensor = self.find_reshaped_vol(img_t1_tensor)
            img_t1ce_tensor = self.find_reshaped_vol(img_t1ce_tensor)
            img_t2_tensor = self.find_reshaped_vol(img_t2_tensor)
            img_flair_tensor = self.find_reshaped_vol(img_flair_tensor)

            self.sub_vol_path = self.root + '/MICCAI_BraTS_2018_Data_Training/generated/visualize/'
            utils.make_dirs(self.sub_vol_path)

            for i in range(len(img_t1_tensor)):
                filename = self.sub_vol_path + 'id_' + str(TEST_SUBJECT) + '_VIZ_' + str(i) + '_'
                f_t1 = filename + 'T1.npy'
                f_t1ce = filename + 'T1CE.npy'
                f_t2 = filename + 'T2.npy'
                f_flair = filename + 'FLAIR.npy'
                f_seg = filename + 'seg.npy'

                np.save(f_t1, img_t1_tensor[i])
                np.save(f_t1ce, img_t1ce_tensor[i])
                np.save(f_t2, img_t2_tensor[i])
                np.save(f_flair, img_flair_tensor[i])
                np.save(f_seg, segmentation_map[i])

                self.full_volume.append(tuple((f_t1, f_t2, f_seg)))
            print("Full validation volume has been generated")
        else:
            self.full_volume = tuple((img_t1_tensor, img_t2_tensor, segmentation_map))

    def find_reshaped_vol(self, tensor):
        tensor = tensor.reshape(-1, self.crop_size[0], self.crop_size[1], self.crop_size[2])
        return tensor

# loader = MICCAIBraTS2018(mode="train", dataset_path="../.././datasets")
