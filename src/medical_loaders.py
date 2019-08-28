import os
import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import nibabel as nib


class MRIDatasetISEG2017(Dataset):
    """
    Code for reading the infant brain MRI dataset of ISEG 2017 challenge
    """
    def __init__(self, mode, dataset_path='./data', dim=(32, 32, 32), fold_id=1, samples=1000):
        """

        :param mode: 'train','val','test'
        :param dataset_path: root dataset folder
        :param dim: subvolume tuple
        :param fold_id: 1 to 10 values
        :param samples: number of sub-volumes that you want to create
        """
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
            # Generates datasets with non-empty sub-volumes!
            self.get_samples()

        elif self.mode == 'val':
            self.list_IDsT1 = val_list_IDsT1
            self.list_IDsT2 = val_list_IDsT2
            self.labels = val_labels
            self.get_samples()

        elif self.mode == 'test':
            self.list_IDsT1 = sorted(glob.glob(os.path.join(self.testing_path, '*T1.img')))
            self.list_IDsT2 = sorted(glob.glob(os.path.join(self.testing_path, '*T2.img')))
            self.labels = None
            if self.debug:
                print(self.list_IDsT1)
        elif self.mode == 'viz':
            self.list_IDsT1 = val_list_IDsT1
            self.list_IDsT2 = val_list_IDsT2
            self.labels = val_labels
            self.get_test_set()


    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        tuple_in = self.list[index]
        img_t1_tensor, img_t2_tensor, segmentation_map = tuple_in
        return img_t1_tensor, img_t2_tensor, segmentation_map


    def get_samples(self):
        total = len(self.list_IDsT1)
        print('total' + self.mode + 'data to generate samples ', total)

        for i in range(self.samples):
            random_index = np.random.randint(total)
            path_T1 = self.list_IDsT1[random_index]
            path_T2 = self.list_IDsT2[random_index]

            while True:
                h_crop = np.random.randint(self.height - self.crop_height)
                w_crop = np.random.randint(self.width - self.crop_width)
                slices = np.random.randint(self.slices - self.crop_slices)

                if self.labels != None:
                    label_path = self.labels[random_index]
                    segmentation_map = self.load_medical_image(label_path, crop=(slices,w_crop,h_crop))

                    if segmentation_map.sum()>1:
                        img_t1_tensor = self.load_medical_image(path_T1, crop=(slices,w_crop,h_crop), type="T1")
                        img_t2_tensor = self.load_medical_image(path_T2, crop=(slices,w_crop,h_crop), type="T2")
                        # visual labels of ISEG-2017
                        label_values = [0, 10, 150, 250]
                        for c, j in enumerate(label_values):
                            segmentation_map[segmentation_map == j] = c
                        break
                    else:
                        continue
                else:
                    segmentation_map = None
                    break

            self.list.append(tuple((img_t1_tensor, img_t2_tensor, segmentation_map)))


    def get_test_set(self):
        total = len(self.list_IDsT1)
        TEST_SUBJECT = 0
        path_T1 = self.list_IDsT1[TEST_SUBJECT]
        path_T2 = self.list_IDsT2[TEST_SUBJECT]
        label_path = self.labels[TEST_SUBJECT]
        # crop = (startSlice, startW,startH)
        crop1 = (45, 0, 0)
        crop2 = (45, 64, 0)
        crop3 = (45, 0, 128)
        crop4 = (45, 64, 128)

        list_crop = [ crop1, crop2, crop3, crop4]
        for crop_idx in list_crop:
            segmentation_map = self.load_medical_image(label_path, crop=crop_idx)
            img_t1_tensor = self.load_medical_image(path_T1, crop=crop_idx, type="T1")
            img_t2_tensor = self.load_medical_image(path_T2, crop=crop_idx, type="T2")
            # visual labels of ISEG-2017
            label_values = [0, 10, 150, 250]
            for c, j in enumerate(label_values):
                segmentation_map[segmentation_map == j] = c

            self.list.append(tuple((img_t1_tensor, img_t2_tensor, segmentation_map)))
        print("DONE volumes =", len(self.list))


    def load_medical_image(self, path, crop, type=None, normalization="mean"):
        slices, w_crop, h_crop = crop

        img = nib.load(path)
        img_np = np.squeeze(img.get_fdata())
        print("ORIG IMG SHAPE",img_np.shape)

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

    def fix_seg_map(self, segmentation_map):
        # visual labels of ISEG-2017
        label_values = [0, 10, 150, 250]
        for c, j in enumerate(label_values):
            segmentation_map[segmentation_map == j] = c
        return segmentation_map


class MRIDatasetMRBRAINS2018(Dataset):
    def __init__(self, mode, dataset_path='./data', dim=(32, 32, 32), fold_id=1, classes=4,samples=1000):
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

        for counter,i in enumerate(self.dirs):
            path_seg = self.training_path + "/" + i
            path_img = self.training_path + "/" + i + "/pre/"

            if (str(counter)==fold_id):
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
        #print('index = ', index)
        #print('self.samples = ', self.samples)
        if index<=self.samples:
            tuple_in = self.list[index]
            return tuple_in
            #img_reg_t1_tensor, img_reg_ir_tensor,img_ir_tensor,img_flair_tensor, segmentation_map = tuple_in
            #return img_t1_tensor, img_t2_tensor, segmentation_map
        else:
            print("E.K.M.EK")
            return None

    def load_medical_image(self, path, crop, type=None, normalization="mean"):
        slices, w_crop, h_crop = crop
        img = nib.load(path)
        img_np = np.squeeze(img.get_fdata()).transpose(2,0,1) # slices are first!!!
        img_np = img_np[slices:slices + self.crop_slices, w_crop:w_crop + self.crop_width,
                 h_crop:h_crop + self.crop_height]
        img_tensor = torch.from_numpy(img_np).float()
        if (type!='label'):
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
        print('total ' + self.mode + ' data to generate samples ', total, 'samples=',self.samples)
        for i in range(self.samples):
            #print('i=',i)
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
                    segmentation_map = self.load_medical_image(label_path, crop=(slices,w_crop,h_crop),type='label')
                    segmentation_map = self.fix_seg_map(segmentation_map)

                    if segmentation_map.sum() > MIN_SEG_VOXELS:
                        img_reg_t1_tensor = self.load_medical_image(path_reg_t1, crop=(slices,w_crop,h_crop),type="T1")
                        img_reg_ir_tensor = self.load_medical_image(path_reg_ir,crop=(slices,w_crop,h_crop), type="reg-IR")
                        img_ir_tensor = self.load_medical_image(path_ir, crop=(slices,w_crop,h_crop),type="IR")
                        img_flair_tensor = self.load_medical_image(path_flair, crop=(slices,w_crop,h_crop),type="FLAIR")
                        break
                    else:
                        continue
                else:
                    segmentation_map = None

                    break
            self.list.append(tuple(
                (img_reg_t1_tensor, img_reg_ir_tensor, img_ir_tensor, img_flair_tensor, segmentation_map)))
            #print('appended')