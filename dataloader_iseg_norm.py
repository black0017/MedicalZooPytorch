import os
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np
import random
import nibabel as nib


class ImageDataset_ISEG(Dataset):
    def __init__(self, mode, dataset_path='./data', dim=(32, 32, 32), fold_id=1, crop=False):
        self.debug = False
        self.mode = mode
        self.root = str(dataset_path)
        self.training_path = self.root + '/iseg/train'
        self.testing_path = self.root + '/iseg/test'
        self.CLASSES = 4
        self.height = 256
        self.width = 192
        self.slices = 144
        self.crop_height = dim[0]
        self.crop_width = dim[1]
        self.crop_slices = dim[2]
        self.fold = str(fold_id)
        self.list = []
        self.crop = crop


        list_IDsT1 = sorted(glob.glob(os.path.join(self.training_path, '*T1.img')))
        list_IDsT2 = sorted(glob.glob(os.path.join(self.training_path, '*T2.img')))
        labels = sorted(glob.glob(os.path.join(self.training_path, '*label.img')))


        training_labels, training_list_IDsT1, training_list_IDsT2, val_list_IDsT1, val_list_IDsT2, val_labels = [], [], [], [], [], []
        print("SELECT Subject with ID {} for Validation".format(self.fold))

        for i in range(len(labels)):
            subject_id = (labels[i].split('/')[-1]).split('-')[1]

            if (subject_id == self.fold):
                val_list_IDsT1.append(list_IDsT1[i])
                val_list_IDsT2.append(list_IDsT2[i])
                val_labels.append(labels[i])
            else:
                training_list_IDsT1.append(list_IDsT1[i])
                training_list_IDsT2.append(list_IDsT2[i])
                training_labels.append(labels[i])


        if (self.mode == 'train'):
            self.list_IDsT1 = training_list_IDsT1
            self.list_IDsT2 = training_list_IDsT2
            self.labels = training_labels

        elif (self.mode == 'val'):
            self.list_IDsT1 = val_list_IDsT1
            self.list_IDsT2 = val_list_IDsT2
            self.labels = val_labels

        elif (self.mode == 'test'):
            self.list_IDsT1 = sorted(glob.glob(os.path.join(self.testing_path, '*T1.img')))
            self.list_IDsT2 = sorted(glob.glob(os.path.join(self.testing_path, '*T2.img')))
            self.labels = None

            if self.debug:
                print(self.list_IDsT1)
    def __len__(self):
        return len(self.list_IDsT1)

    def __getitem__(self, index):

        path_T1 = self.list_IDsT1[index]
        path_T2 = self.list_IDsT2[index]
        img_t1_tensor = self.load_medical_image(path_T1, "T1", crop=self.crop)
        img_t2_tensor = self.load_medical_image(path_T2, "T2", crop=self.crop)

        if (self.labels != None):
            label_path = self.labels[index]

            if (self.debug):
                print("-------")
                print("PATHS for index : ", index)
                print(path_T1)
                print(path_T2)
                print(label_path)
                print("-------")

            segmentation_map = self.load_medical_image(label_path,crop=crop)
            label_values = [0, 10, 150, 250]  # visual labels of ISEG-2017
            for c, j in enumerate(label_values):
                segmentation_map[segmentation_map == j] = c

            segmentation_map = one_hot(segmentation_map, self.CLASSES)
        else:
            segmentation_map = None

        return img_t1_tensor, img_t2_tensor, segmentation_map

    def get_samples(self, n=100):
        crop = True
        total = len(self.list_IDsT1)
        for i in range(n):
            random_index = np.random.randint(total)
            path_T1 = self.list_IDsT1[random_index]
            path_T2 = self.list_IDsT2[random_index]

            img_t1_tensor = self.load_medical_image(path_T1, "T1", crop=crop)
            img_t2_tensor = self.load_medical_image(path_T2, "T2", crop=crop)

            if (self.labels != None):
                label_path = self.labels[random_index]
                if self.debug:
                    print("-------")
                    print("PATHS for index : ", random_index)
                    print(label_path)

                segmentation_map = self.load_medical_image(label_path,crop=crop)
                label_values = [0, 10, 150, 250]  # visual labels of ISEG-2017
                for c, j in enumerate(label_values):
                    segmentation_map[segmentation_map == j] = c
                segmentation_map = one_hot(segmentation_map, self.CLASSES)
            else:
                segmentation_map = None

            self.list.append(tuple((img_t1_tensor, img_t2_tensor, segmentation_map)))


    def load_medical_image(self, path, type=None, crop=False):

        crop_height = self.crop_height
        crop_width = self.crop_width
        crop_slices = self.crop_slices

        h_crop = np.random.randint(self.height - crop_height)
        w_crop = np.random.randint(self.width - crop_width)
        slices = np.random.randint(self.slices - crop_slices)

        img = nib.load(path)
        img_np = np.squeeze(img.get_fdata())
        """
        print("INPUT shpe",type)
        print(img_np.shape)
        """
        if crop:
            img_np = img_np[slices:slices + crop_slices, w_crop:w_crop + crop_width,h_crop:h_crop + crop_height]
        img_tensor = torch.from_numpy(img_np).float()

        if (type == "T1"):
            #img_tensor = (img_tensor - self.normMu_t1) / self.normSigma_t1
            img_tensor = img_tensor/250.0
            return img_tensor.unsqueeze(0)
        elif (type == "T2"):
            #img_tensor = (img_tensor - self.normMu_t2) / self.normSigma_t2
            img_tensor = img_tensor / 250.0
            return img_tensor.unsqueeze(0)
        else:
            if self.debug:
                print('MAX =  ', np.max(img_np))
                print('Min =  ', np.min(img_np))
                print(np.unique(img_np))
            return img_tensor




class ImageDataset_ISEG_sample(Dataset):
    def __init__(self, mode, dataset_path='./data', dim=(32, 32, 32), fold_id=1, samples=1000):
        self.debug = False
        self.mode = mode
        self.root = str(dataset_path)
        self.training_path = self.root + '/iseg/train'
        self.testing_path = self.root + '/iseg/test'
        self.CLASSES = 4
        self.height = 256
        self.width = 192
        self.slices = 144
        self.crop_height = dim[0]
        self.crop_width = dim[1]
        self.crop_slices = dim[2]
        self.fold = str(fold_id)
        self.list = []
        self.samples = samples

        list_IDsT1 = sorted(glob.glob(os.path.join(self.training_path, '*T1.img')))
        list_IDsT2 = sorted(glob.glob(os.path.join(self.training_path, '*T2.img')))
        labels = sorted(glob.glob(os.path.join(self.training_path, '*label.img')))

        training_labels, training_list_IDsT1, training_list_IDsT2, val_list_IDsT1, val_list_IDsT2, val_labels = [], [], [], [], [], []
        print("SELECT Subject with ID {} for Validation".format(self.fold))

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

        elif self.mode == 'val':
            self.list_IDsT1 = val_list_IDsT1
            self.list_IDsT2 = val_list_IDsT2
            self.labels = val_labels

        elif self.mode == 'test':
            self.list_IDsT1 = sorted(glob.glob(os.path.join(self.testing_path, '*T1.img')))
            self.list_IDsT2 = sorted(glob.glob(os.path.join(self.testing_path, '*T2.img')))
            self.labels = None

            if self.debug:
                print(self.list_IDsT1)

        # global stats
        self.normMu_t1 = 31.6907
        self.normMu_t2 = 38.0572
        self.normSigma_t1 = 91.4297
        self.normSigma_t2 = 112.3855
        self.get_samples()

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        if index<=self.samples:
            tuple_in = self.list[index]
            img_t1_tensor, img_t2_tensor, segmentation_map = tuple_in
            return img_t1_tensor, img_t2_tensor, segmentation_map
        else:
            print("E.K.M.EK")
            return None

    def get_samples(self):
        crop = True
        total = len(self.list_IDsT1)
        print('total data to gen samples ',total)
        for i in range(self.samples):
            random_index = np.random.randint(total)
            path_T1 = self.list_IDsT1[random_index]
            path_T2 = self.list_IDsT2[random_index]

            h_crop = np.random.randint(self.height - self.crop_height)
            w_crop = np.random.randint(self.width - self.crop_width)
            slices = np.random.randint(self.slices - self.crop_slices)

            img_t1_tensor = self.load_medical_image(path_T1, crop=(slices,w_crop,h_crop), type="T1")
            img_t2_tensor = self.load_medical_image(path_T2, crop=(slices,w_crop,h_crop), type="T2")

            if self.labels != None:
                label_path = self.labels[random_index]
                segmentation_map = self.load_medical_image(label_path, crop=(slices,w_crop,h_crop) )
                label_values = [0, 10, 150, 250]  # visual labels of ISEG-2017
                for c, j in enumerate(label_values):
                    segmentation_map[segmentation_map == j] = c
            else:
                segmentation_map = None

            self.list.append(tuple((img_t1_tensor, img_t2_tensor, segmentation_map)))

    def load_medical_image(self, path,crop, type=None, normalization="mean"):
        slices, w_crop, h_crop = crop

        img = nib.load(path)
        img_np = np.squeeze(img.get_fdata())

        if crop:
            img_np = img_np[slices:slices + self.crop_slices , w_crop:w_crop + self.crop_width,h_crop:h_crop + self.crop_height]
        img_tensor = torch.from_numpy(img_np).float()

        if (type == "T1"):
            if normalization=="mean":
                mask = img_tensor.ne(0.0)
                desired = img_tensor[mask]
                mean_val, std_val = desired.mean(), desired.std()
                img_tensor = (img_tensor-mean_val)/std_val
            else:
                img_tensor = img_tensor / 250.0

            return img_tensor.unsqueeze(0)

        elif (type == "T2"):
            if normalization == "mean":
                mask = img_tensor.ne(0.0)
                desired = img_tensor[mask]
                mean_val, std_val = desired.mean(), desired.std()
                img_tensor = (img_tensor - mean_val) / std_val
            else:
                img_tensor = img_tensor / 250.0

            return img_tensor.unsqueeze(0)
        else:
            return img_tensor


def one_hot(labels, num_classes):
    labels = labels.reshape(-1, 1).float()
    desired = torch.arange(num_classes).reshape(1, num_classes).float()
    one_hot_target = (labels == desired).float()
    return one_hot_target