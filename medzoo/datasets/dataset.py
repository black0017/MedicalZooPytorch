

from abc import ABC, abstractmethod

from torch.utils.data import Dataset


"""
Based on this repository: https://github.com/black0017/MICCAI-2019-Prostate-Cancer-segmentation-challenge
"""


class MedzooDataset(Dataset):

    def __init__(self, config, mode, root_path='.././datasets'):
        """
        Args:
        """
        config = config.config_dataset
        self.mode = mode
        self.root_path = root_path
        self.crop_size = config.crop_size
        self.full_vol_dim = config.image_dim
        self.classes = config.classes
        self.threshold = config.threshold
        self.split = config.split
        self.augmentation = config.augmentation
        self.normalization = config.normalization
        self.subvol = '_vol_' + str(self.crop_size[0]) + 'x' + str(self.crop_size[1]) + 'x' + str(self.crop_size[2])
        self.affine = None
        self.augment_transform = None
        self.samples = config[self.mode].total_samples
        self.fold = int(config.fold)
        self.save = config.save
        self.modalities = config.modalities
        self.voxels_space = config.voxels_space
        self.to_canonical = config.to_canonical
        self.transform = None
        self.split_idx = config.split_idx



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
        elif self.mode == 'test':
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
    def preprocess_viz(self):
        pass

    @abstractmethod
    def save(self):
        pass





