import glob

import numpy as np
import torch
from torch.utils.data import Dataset

import medzoo.utils as utils
from medzoo.common.medloaders import medical_image_process as img_loader
from medzoo.datasets.dataset import MedzooDataset

"""
Based on this repository: https://github.com/black0017/MICCAI-2019-Prostate-Cancer-segmentation-challenge
"""


class MICCAI2019_gleason_pathology(MedzooDataset):
    """
    Code for reading Gleason 2019 MICCAI Challenge
    """

    def __init__(self, config, mode, dataset_path='.././datasets'):
        """
        Args:
            mode: 'train','val'
            image_paths: image dataset paths
            label_paths: label dataset paths
            crop_dim: 2 element tuple to decide crop values
            samples: number of sub-grids to create(patches of the input img)
        """
        super().__init__(config, mode, dataset_path)

        self.image_paths = sorted(glob.glob(dataset_path + "/MICCAI_2019_pathology_challenge/Train Imgs/Train Imgs/*.jpg"))
        self.label_paths = sorted(glob.glob(dataset_path + "/MICCAI_2019_pathology_challenge/Labels/*.png"))

        self.image_paths, self.abel_paths = utils.shuffle_lists(self.image_paths, self.label_paths, seed=17)
        self.full_volume = None

        self.slices = 244  # dataset instances
        self.sample_list = []
        self.per_image_sample = int(self.samples / self.slices)
        if self.per_image_sample < 1:
            self.per_image_sample = 1

        print("per image sampleeeeee", self.per_image_sample)

        sub_grid = '_2dgrid_' + str(self.crop_size[0]) + 'x' + str(self.crop_size[1])

        self.sub_vol_path = self.root_path + '/MICCAI_2019_pathology_challenge/generated/' + mode + sub_grid + '/'

        self.list_imgs = None
        self.list_labels = None

        self.split_idx = 150

        self.load_dataset()



    def preprocess_train(self):
        self.list_imgs = self.image_paths[0:self.split_idx]
        self.list_labels = self.label_paths[0:self.split_idx]

    def preprocess_val(self):
        self.list_imgs = self.image_paths[self.split_idx:]
        self.list_labels = self.label_paths[self.split_idx:]

    def save(self):
        utils.make_dirs(self.sub_vol_path)



    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        if self.save:
            img_path, seg_path = self.sample_list[index]
            return np.load(img_path), np.load(seg_path)
        else:
            tuple_in = self.sample_list[index]
            img_tensor, segmentation_map = tuple_in
            return img_tensor, segmentation_map

    def preprocess(self):
        """

        """
        total_imgs = len(self.list_imgs)
        print('Total ' + self.mode + ' data to generate samples:', total_imgs)
        print('Mode: ' + self.mode + ' 2d sub grids samples to generate: ', self.samples, ' Input images: ', total_imgs)
        for j in range(total_imgs):
            input_path = self.list_imgs[j]
            label_path = self.list_labels[j]
            img_numpy = img_loader.load_2d_image(input_path, type="RGB")
            label_numpy = img_loader.load_2d_image(label_path, type='LA')
            for i in range(self.per_image_sample):
                h_crop, w_crop = self.generate_patch(img_numpy)
                img_cropped = img_numpy[h_crop:(h_crop + self.crop_size[0]),
                              w_crop:(w_crop + self.crop_size[1]), :]

                label_cropped = label_numpy[h_crop:(h_crop + self.crop_size[0]),
                                w_crop:(w_crop + self.crop_size[1])]

                img_tensor = torch.from_numpy(img_cropped).float()
                label_tensor = torch.from_numpy(label_cropped)

                img_tensor = img_tensor.permute(2, 0, 1)
                img_tensor = self.norm_img(img_tensor)

                if self.save:
                    filename = self.sub_vol_path + 'id_' + str(j) + '_s_' + str(i) + '_'
                    f_1 = filename + 'input.npy'
                    f_2 = filename + 'label.npy'
                    np.save(f_1, img_tensor)
                    np.save(f_2, label_tensor)
                    self.sample_list.append(tuple((f_1, f_2)))
                else:
                    self.sample_list.append(tuple((img_tensor, label_tensor)))

    def generate_patch(self, img):
        """

        Args:
            img:

        Returns:

        """
        h, w, c = img.shape
        if h < self.crop_size[0] or w < self.crop_size[1]:
            print('dim error')
            print(h, self.crop_size[0], w, self.crop_size[1])
        h_crop = np.random.randint(h - self.crop_size[0])
        w_crop = np.random.randint(w - self.crop_size[1])
        return h_crop, w_crop

    def norm_img(self, img_tensor):
        """

        Args:
            img_tensor:

        Returns:

        """
        mask = img_tensor.ne(0.0)
        desired = img_tensor[mask]
        mean_val, std_val = desired.mean(), desired.std()
        img_tensor = (img_tensor - mean_val) / std_val
        return img_tensor

    def generate_train_labels(self):
        """

        """
        tuple_maps = read_labels(self.root_path)
        preprocess_labels(tuple_maps)


def check_path_in_list(path, list):
    """
    Checks a path if exist in the other list
    """
    key = path.split('/')[-1]
    for full_path in list:
        path_id = full_path.split('/')[-1]
        if path_id == key:
            image_numpy = img_loader.load_2d_image(full_path, type="LA")
            return image_numpy
        return None


def get_majority_vote(a):
    """
    Returns the majority vote element of a list
    """
    return max(map(lambda val: (a.count(val), val), set(a)))[1]


def vote(stacked_labels):
    """
    Performs majority voting on the stacked labels
    """
    voters, height, width = stacked_labels.shape
    final_labels = stacked_labels.sum(axis=0)

    for i in range(height):
        for j in range(width):
            votes = stacked_labels[:, i, j]
            value = get_majority_vote(votes.tolist())
            final_labels[i, j] = value
    print('original: ', np.unique(stacked_labels), 'voted: ', np.unique(final_labels))
    return final_labels


def preprocess_labels(maps_tuple):
    """
    Majority labeling vote to produce ground truth labels
    """
    label_list = []

    m1, m2, m3, m4, m5, m6 = maps_tuple
    for j in range(len(m5)):
        path = m5[j]  # as a reference annotation

        key = path.split('/')[-1]

        image_list = []
        # voter 5
        image_list.append(img_loader.load_2d_image(path, type="LA"))

        # voter 1
        image = check_path_in_list(path, m1)
        if image is not None:
            image_list.append(image)

        # voter 2
        image = check_path_in_list(path, m2)
        if image is not None:
            image_list.append(image)

        # voter 3
        image = check_path_in_list(path, m3)
        if image is not None:
            image_list.append(image)

        # voter 4
        image = check_path_in_list(path, m4)
        if image is not None:
            image_list.append(image)

        # voter 6
        image = check_path_in_list(path, m6)
        if image is not None:
            image_list.append(image)

        stacked_labels = np.stack(image_list, axis=0)
        label = vote(stacked_labels)
        # Todo pillow image save here
        # imageio.imwrite('./labels/' + key, label.astype('uint8'))


def read_labels(root_path):
    """
    Reads labels and returns them in a tuple of sorted lists
    """
    label_list = []
    map_1 = sorted(glob.glob(root_path + 'Maps1_T/Maps1_T/*.png'))
    map_2 = sorted(glob.glob(root_path + 'Maps2_T/Maps2_T/*.png'))
    map_3 = sorted(glob.glob(root_path + 'Maps3_T/Maps3_T/*.png'))
    map_4 = sorted(glob.glob(root_path + 'Maps4_T/Maps4_T/*.png'))
    map_5 = sorted(glob.glob(root_path + 'Maps5_T/Maps5_T/*.png'))
    map_6 = sorted(glob.glob(root_path + 'Maps6_T/Maps6_T/*.png'))
    return map_1, map_2, map_3, map_4, map_5, map_6
