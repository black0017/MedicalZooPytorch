import os
import torch
from torch.utils.data import Dataset
import glob
import numpy as np
from utils import read_filepaths
from PIL import Image
from torchvision import transforms

COVIDxDICT = {'pneumonia': 0, 'normal': 1, 'COVID-19': 2}


class COVIDxDataset(Dataset):
    """
    Code for reading the COVIDxDataset
    """

    def __init__(self, mode, n_classes=3, dataset_path='./datasets', dim=(224, 224)):
        self.root = str(dataset_path) + '/' + mode + '/'

        self.CLASSES = n_classes
        self.dim = dim
        self.COVIDxDICT = {'pneumonia': 0, 'normal': 1, 'COVID-19': 2}
        testfile = './test_split_v2.txt'
        trainfile = './train_split_v2.txt'
        if (mode == 'train'):
            self.paths, self.labels = read_filepaths(trainfile)
        elif (mode == 'test'):
            self.paths, self.labels = read_filepaths(testfile)
        print("{} examples =  {}".format(mode, len(self.paths)))
        self.mode = mode

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        image_tensor = self.load_image(self.root + self.paths[index], self.dim, augmentation=self.mode)
        label_tensor = torch.tensor(self.COVIDxDICT[self.labels[index]], dtype=torch.long)

        return image_tensor, label_tensor

    def load_image(self, img_path, dim, augmentation='test'):
        if not os.path.exists(img_path):
            print("IMAGE DOES NOT EXIST {}".format(img_path))
        image = Image.open(img_path).convert('RGB')
        image = image.resize(dim)

        t = transforms.ToTensor()

        norm = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[1, 1, 1])

        image_tensor = norm(t(image))

        return image_tensor