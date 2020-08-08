

from abc import ABC, abstractmethod

from torch.utils.data import Dataset


"""
Based on this repository: https://github.com/black0017/MICCAI-2019-Prostate-Cancer-segmentation-challenge
"""


class MedzooDataset(ABC, Dataset):

    def __init__(self, config, mode, root_path='.././datasets'):
        """
        Args:
        """

        self.mode = mode
        self.root_path = root_path
        self.crop_dim = config.dim
        self.full_vol_dim = config.full_dim
        self.classes = config.classes
        self.threshold = config.threshold
        self.split = config.split
        self.load = config.loadData
        self.save = config.loadData
        self.augmentation = config.augmentation
        self.normalization = config.normalization
        self.subvol = '_vol_' + str(self.crop_dim[0]) + 'x' + str(self.crop_dim[1]) + 'x' + str(self.crop_dim[2])
        self.affine = None
        self.augment_transform = None

        if mode == 'train':
            self.samples = config.samples_train
        else:
            self.samples = config.samples_val

    def load_dataset(self):

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





