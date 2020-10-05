from abc import abstractmethod

import torch
from torch.utils.data import Dataset


class SimpleDataset(Dataset):

    def __init__(self, config, mode, root_path='.././datasets'):
        """
        Args:
        """

        self.mode = mode
        self.root_path = root_path
        self.crop_size = config.crop_size
        self.full_vol_dim = config.image_dim
        self.classes = config.classes
        self.threshold = config.threshold
        self.split = config.split
        self.augmentation = config.augmentation
        self.normalization = config.normalization
        self.samples = config[self.mode].total_samples
        self.num_modalities = config.num_modalities
        self.subvol = '_vol_' + str(self.crop_size[0]) + 'x' + str(self.crop_size[1]) + 'x' + str(self.crop_size[2])
        self.affine = None
        self.augment_transform = None
        self.full_volume = None

        self.modality_keys = ['T1', 'T2', 'label']  # config.modality_keys#
        self.dict = {}

    def __len__(self):
        return len(self.dict)

    def __getitem__(self, index):
        inputs = []
        sample = self.dict[index]
        for i in range(self.num_modalities - 1):
            x = self.load_data(i)
            inputs.append(x)
        inputs = torch.stack(inputs)

        y = self.load_data(self.num_modalities - 1)
        return inputs, y

    def load_data(self, path):

        if self.load:
            self.load()
            return

        self.preprocess_dataset()

    def preprocess_dataset(self):

        self.preprocess()

        if self.augmentation:
            self.augment()

        if self.mode == 'train':
            self.preprocess_train()
        elif self.mode == 'val':
            self.preprocess_val()
        else:
            self.preprocess_test()

        if self.save:
            self.save()

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def preprocess(self):
        pass

    @abstractmethod
    def augment(self):
        pass

    @abstractmethod
    def preprocess_train(self):
        pass

    @abstractmethod
    def preprocess_val(self):
        pass

    @abstractmethod
    def preprocess_test(self):
        pass

    @abstractmethod
    def save(self):
        pass
