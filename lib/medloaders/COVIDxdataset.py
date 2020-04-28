import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms


from lib.medloaders import medical_image_process as img_loader


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
        testfile = '../datasets/covid_x_dataset/test_split_v2.txt'
        trainfile = '../datasets/covid_x_dataset/train_split_v2.txt'
        if (mode == 'train'):
            self.paths, self.labels = read_filepaths(trainfile)
        elif (mode == 'val'):
            self.paths, self.labels = read_filepaths(testfile)
        print("{} examples =  {}".format(mode, len(self.paths)))
        self.mode = mode
        self.full_volume = None
        self.affine = None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        image_tensor = self.load_image(self.root + self.paths[index], self.dim, augmentation=self.mode)
        label_tensor = torch.tensor(self.COVIDxDICT[self.labels[index]], dtype=torch.long)

        return image_tensor, label_tensor

    def load_image(self, img_path, resize_dim):
        if not os.path.exists(img_path):
            print("IMAGE DOES NOT EXIST {}".format(img_path))
        image = img_loader.load_2d_image(img_path, resize_dim)
        t = transforms.ToTensor()
        norm = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[1, 1, 1])
        return norm(t(image))


def read_filepaths(file):
    paths, labels = [], []
    with open(file, 'r') as f:
        lines = f.read().splitlines()

        for idx, line in enumerate(lines):
            if ('/ c o' in line):
                break
            subjid, path, label = line.split(' ')

            paths.append(path)
            labels.append(label)
    return paths, labels
