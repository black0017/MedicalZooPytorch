import os

import torch
import torchvision.transforms as transforms
from PIL import Image

from medzoo.datasets.dataset import MedzooDataset
from medzoo.utils.covid_utils import read_txt


class CovidCTDataset(MedzooDataset):
    """

    """
    def __init__(self, config, mode, dataset_path, txt_COVID, txt_NonCOVID, transform=None):
        """
        Args:
            txt_path (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        File structure:
        - root_dir
            - CT_COVID
                - img1.png
                - img2.png
                - ......
            - CT_NonCOVID
                - img1.png
                - img2.png
                - ......
        """
        super().__init__(config, mode, dataset_path)

        self.txt_path = [txt_COVID, txt_NonCOVID]
        self.classes_names = ['CT_COVID', 'CT_NonCOVID']
        self.classes = len(self.classes_names)
        self.img_list = []
        for c in range(self.classes):
            cls_list = [[os.path.join(self.root_path, self.classes_names[c], item), c] for item in read_txt(self.txt_path[c])]
            self.img_list += cls_list

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = None

    def preprocess_train(self):
        train_transformer = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop((224), scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize
        ])
        self.transform = train_transformer

    def preprocess_val(self):
        val_transformer = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            self.normalize
        ])
        self.transform = val_transformer


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_list[idx][0]

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(int(self.img_list[idx][1]), dtype=torch.long)
