import os
from torch.utils.data import Dataset
import glob
import numpy as np

from src.medloaders import img_loader
import src.utils as utils


class MRIDatasetISEG2017(Dataset):
    """
    Code for reading the infant brain MRI dataset of ISEG 2017 challenge
    """
    def __init__(self, mode, dataset_path='./datasets', dim=(32, 32, 32), fold_id=1, samples=1000, save=True):
        """
        :param mode: 'train','val','test'
        :param dataset_path: root dataset folder
        :param dim: subvolume tuple
        :param fold_id: 1 to 10 values
        :param samples: number of sub-volumes that you want to create
        """
        self.mode = mode
        self.root = str(dataset_path)
        self.training_path = self.root + '/iseg_2017/iSeg-2017-Training/'
        self.testing_path = self.root + '/iseg_2017/iSeg-2017-Testing/'
        self.save = save
        self.CLASSES = 4
        self.full_vol_dim = (144, 192, 256)  # slice, width, height
        self.crop_size = dim
        self.fold = str(fold_id)
        self.list = []
        self.samples = samples
        self.full_volume = None


        subvol = '_vol_' + str(dim[0]) + 'x' + str(dim[1]) + 'x' + str(dim[2])

        if self.save:
            self.sub_vol_path = self.root + '/iseg_2017/generated/' + mode + subvol + '/'
            utils.make_dirs(self.sub_vol_path)

        list_IDsT1 = sorted(glob.glob(os.path.join(self.training_path, '*T1.img')))
        list_IDsT2 = sorted(glob.glob(os.path.join(self.training_path, '*T2.img')))
        labels = sorted(glob.glob(os.path.join(self.training_path, '*label.img')))
        self.affine = img_loader.load_affine_matrix(list_IDsT1[0])

        training_labels, training_list_IDsT1, training_list_IDsT2, val_list_IDsT1, val_list_IDsT2, val_labels = [], [], [], [], [], []
        print("SELECT subject with ID {} for Validation".format(self.fold))

        for i in range(len(labels)):
            subject_id = (labels[i].split('/')[-1]).split('-')[1]

            if subject_id == self.fold:
                val_list_IDsT1.append(list_IDsT1[i])
                val_list_IDsT2.append(list_IDsT2[i])
                val_labels.append(labels[i])
            else:
                training_list_IDsT1.append(list_IDsT1[i])
                training_list_IDsT2.append(list_IDsT2[i])
                training_labels.append(labels[i])

        if self.mode == 'train':
            self.list_IDsT1 = training_list_IDsT1
            self.list_IDsT2 = training_list_IDsT2
            self.labels = training_labels
            # Generates datasets with non-empty sub-volumes!
            self.get_samples()

        elif self.mode == 'val':
            self.list_IDsT1 = val_list_IDsT1
            self.list_IDsT2 = val_list_IDsT2
            self.labels = val_labels
            self.get_samples()
            self.get_viz_set()

        elif self.mode == 'test':
            self.list_IDsT1 = sorted(glob.glob(os.path.join(self.testing_path, '*T1.img')))
            self.list_IDsT2 = sorted(glob.glob(os.path.join(self.testing_path, '*T2.img')))
            self.labels = None

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        # offline data
        if self.save:
            t1_path, t2_path, seg_path = self.list[index]
            return np.load(t1_path), np.load(t2_path), np.load(seg_path)
        # on-memory saved data
        else:
            return self.list[index]

    def get_samples(self):
        total = len(self.list_IDsT1)
        TH = 160  # threshold for non empty volumes
        print('Mode: ' + self.mode + ' Subvolume samples to generate: ', self.samples, ' Volumes: ', total)

        for i in range(self.samples):
            random_index = np.random.randint(total)
            path_T1 = self.list_IDsT1[random_index]
            path_T2 = self.list_IDsT2[random_index]

            while True:
                slices = np.random.randint(self.full_vol_dim[0] - self.crop_size[0])
                w_crop = np.random.randint(self.full_vol_dim[1] - self.crop_size[1])
                h_crop = np.random.randint(self.full_vol_dim[2] - self.crop_size[2])

                if self.labels is not None:
                    label_path = self.labels[random_index]
                    segmentation_map = img_loader.load_medical_image(label_path, crop_size=self.crop_size,
                                                                     crop=(slices, w_crop, h_crop), type='label')
                    if segmentation_map.sum() > TH:
                        img_t1_tensor = img_loader.load_medical_image(path_T1, crop_size=self.crop_size,
                                                                      crop=(slices, w_crop, h_crop), type="T1")
                        img_t2_tensor = img_loader.load_medical_image(path_T2, crop_size=self.crop_size,
                                                                      crop=(slices, w_crop, h_crop), type="T2")
                        segmentation_map = self.fix_seg_map(segmentation_map)
                        break
                    else:
                        continue
                else:
                    segmentation_map = None
                    break

            if self.save:

                filename = self.sub_vol_path + 'id_' + str(random_index) + '_s_' + str(i) + '_'
                f_t1 = filename + 'T1.npy'
                f_t2 = filename + 'T2.npy'
                f_seg = filename + 'seg.npy'
                np.save(f_t1, img_t1_tensor)
                np.save(f_t2, img_t2_tensor)
                np.save(f_seg, segmentation_map)
                self.list.append(tuple((f_t1, f_t2, f_seg)))
            else:
                self.list.append(tuple((img_t1_tensor, img_t2_tensor, segmentation_map)))

    def get_viz_set(self):
        """
        Returns total 3d input volumes(t1 and t2) and segmentation maps
        3d total vol shape : torch.Size([1, 144, 192, 256])
        :return:
        """
        TEST_SUBJECT = 0
        path_T1 = self.list_IDsT1[TEST_SUBJECT]
        path_T2 = self.list_IDsT2[TEST_SUBJECT]
        label_path = self.labels[TEST_SUBJECT]

        segmentation_map = img_loader.load_medical_image(label_path, viz3d=True)

        img_t1_tensor = img_loader.load_medical_image(path_T1, type="T1", viz3d=True)
        img_t2_tensor = img_loader.load_medical_image(path_T2, type="T2", viz3d=True)
        segmentation_map = self.fix_seg_map(segmentation_map)



        ### TO DO SAVE FULL VOLUME AS numpy

        if self.save:
            self.full_volume = []

            segmentation_map = segmentation_map.reshape(-1, self.crop_size[0], self.crop_size[1], self.crop_size[2])
            img_t1_tensor = img_t1_tensor.reshape(-1, self.crop_size[0], self.crop_size[1], self.crop_size[2])
            img_t2_tensor = img_t1_tensor.reshape(-1, self.crop_size[0], self.crop_size[1], self.crop_size[2])
            self.sub_vol_path = self.root + '/iseg_2017/generated/visualize/'
            utils.make_dirs(self.sub_vol_path)



            for i in range(len(img_t1_tensor)):

                filename = self.sub_vol_path + 'id_' + str(TEST_SUBJECT) + '_VIZ_' + str(i) + '_'
                f_t1 = filename + 'T1.npy'
                f_t2 = filename + 'T2.npy'
                f_seg = filename + 'seg.npy'

                np.save(f_t1, img_t1_tensor[i])
                np.save(f_t2, img_t2_tensor[i])

                np.save(f_seg, segmentation_map[i])
                self.full_volume.append(tuple((f_t1, f_t2, f_seg)))
            print("Full validation volume has been generated")
        else:
            self.full_volume = tuple((img_t1_tensor, img_t2_tensor, segmentation_map))


    def fix_seg_map(self, segmentation_map):
        # visual labels of ISEG-2017
        label_values = [0, 10, 150, 250]
        for c, j in enumerate(label_values):
            segmentation_map[segmentation_map == j] = c
        return segmentation_map
