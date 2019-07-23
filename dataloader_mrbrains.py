import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import nibabel as nib


def find_global_stats(training_generator):
    print("LEN = ", len(training_generator))

    mean_t1 = []
    mean_reg_ir = []
    mean_ir = []
    mean_flair = []

    std_t1 = []
    std_ir = []
    std_reg_ir = []
    std_flair = []

    for i, tu in enumerate(training_generator):
        img_reg_t1_tensor, img_reg_ir_tensor,img_ir_tensor,img_flair_tensor, segmentation_map = tu

        mean_t1.append(torch.mean(img_reg_t1_tensor))
        mean_reg_ir.append(torch.mean(img_reg_ir_tensor))
        mean_ir.append(torch.mean(img_ir_tensor))
        mean_flair.append(torch.mean(img_flair_tensor))
        print(mean_t1)

        std_t1.append(torch.std(img_reg_t1_tensor))
        std_reg_ir.append(torch.std(img_reg_ir_tensor))
        std_ir.append(torch.std(img_ir_tensor))
        std_flair.append(torch.std(img_flair_tensor))

    m1 = torch.stack(mean_t1).mean(dim=0)
    m2 = torch.stack(mean_reg_ir).mean(dim=0)
    m3 = torch.stack(mean_ir).mean(dim=0)
    m4 = torch.stack(mean_flair).mean(dim=0)

    std_1 = torch.stack(std_t1).mean(dim=0)
    std_2 = torch.stack(std_reg_ir).mean(dim=0)
    std_3 = torch.stack(std_ir).mean(dim=0)
    std_4 = torch.stack(std_flair).mean(dim=0)

    print("mean T1 intensity", m1.item())
    print("mean r ir intensity", m2.item())
    print("mean ir intensity", m3.item())
    print("mean T1 intensity", m4.item())

    print("STD 1", std_1)
    print("STD 2", std_2)
    print("STD 3", std_3)
    print("STD 4", std_4)

    print("DONE")


class ImageDataset_mrbrains(Dataset):
    def __init__(self, mode, dataset_path='./data', dim=(32, 32, 32), fold_id=1, crop=False):
        self.debug = False
        self.mode = mode
        self.root = dataset_path
        self.training_path = self.root + '/mrbrains'
        self.dirs = os.listdir(self.training_path)
        self.crop = crop
        self.CLASSES = 9

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


        # todo find global stats
        self.mean_t1 = 43.526763916015625
        self.mean_reg_ir = 1841.6517333984375
        self.mean_ir = 1737.3948974609375
        self.mean_flair = 59.16896438598633

        self.std_t1 = 69.1425
        self.std_reg_ir = 123.9639
        self.std_ir = 103.8026
        self.std_flair = 96.9124

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

    def __len__(self):
        return len(self.list_t1)

    def __getitem__(self, index):

        label_path = self.labels[index]
        path_flair = self.list_flair[index]
        path_ir = self.list_ir[index]
        path_reg_ir = self.list_reg_ir[index]
        path_reg_t1 = self.list_reg_t1[index]

        if (self.debug):
            print("-------")
            print("PATHS for index : ", index)
            print(label_path)
            print(path_reg_ir)
            print(path_reg_t1)
            print("-------")

        img_reg_t1_tensor = self.load_medical_image(path_reg_t1, "T1")
        img_reg_ir_tensor = self.load_medical_image(path_reg_ir, "reg-IR")
        img_ir_tensor = self.load_medical_image(path_ir, "IR")
        img_flair_tensor = self.load_medical_image(path_flair, "FLAIR")

        segmentation_map = self.load_medical_image(label_path, "label")
        segmentation_map = one_hot(segmentation_map, self.CLASSES)

        return img_reg_t1_tensor, img_reg_ir_tensor,img_ir_tensor,img_flair_tensor, segmentation_map

    def load_medical_image(self, path, type=None):

        crop_height = self.crop_height
        crop_width = self.crop_width
        crop_slices = self.crop_slices

        h_crop = np.random.randint(self.height - crop_height)
        w_crop = np.random.randint(self.width - crop_width)
        slices = np.random.randint(self.slices - crop_slices)

        img = nib.load(path)
        img_np = np.squeeze(img.get_fdata()).transpose(2,0,1) # slices are first

        if (type=='label'):
            #print('unique labels:')
            img_np[img_np >8 ] = 3
            #print(np.unique(img_np))

        if self.crop:
            print("shape before crop", img_np.shape)
            img_np = img_np[slices:slices + crop_slices, w_crop:w_crop + crop_width, h_crop:h_crop + crop_height]

        img_tensor = torch.from_numpy(img_np).float()

        if type == 'T1':
            img_tensor = (img_tensor-self.mean_t1)/self.std_t1
        elif type == 'reg-IR':
            img_tensor = (img_tensor - self.mean_reg_ir) / self.std_reg_ir
        elif type == 'IR':
            img_tensor = (img_tensor - self.mean_ir) / self.std_ir
        elif type == 'FLAIR':
            img_tensor = (img_tensor - self.mean_flair) / self.std_flair

        return img_tensor


def one_hot(labels, num_classes):
    labels = labels.reshape(-1, 1).float()
    desired = torch.arange(num_classes).reshape(1, num_classes).float()
    one_hot_target = (labels == desired).float()

    return one_hot_target