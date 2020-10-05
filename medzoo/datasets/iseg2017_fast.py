from omegaconf import OmegaConf

from medzoo.common.augment3D import Compose, MRIReader, DictToNumpy, ScaleIntensity, RandomCrop, DictToTensor
from medzoo.datasets.simple_dataset import SimpleDataset


class MRIDatasetISEG2017Fast(SimpleDataset):
    """
    Code for reading the infant brain MRI dataset of ISEG 2017 challenge
    """

    def __init__(self, mode, dataset_path='./datasets'):
        """

        Args:
            mode: 'train','val','test'
            dataset_path: root dataset folder
            crop_dim: subvolume tuple
            fold_id: 1 to 10 values
            samples: number of sub-volumes that you want to create
        """
        config = OmegaConf.load(
            '/mnt/784C5F3A4C5EF1FC/PROJECTS/MedZoo_dev/MedicalZooPytorch/medzoo/datasets/iseg_2017/defaults.yaml')[
            'dataset_config']

        super().__init__(config, mode, root_path=dataset_path)
        self.mode = mode
        self.root = str(dataset_path)

        import json
        with open('../medzoo/datasets/iseg_2017/files/iseg2017train.json', 'r') as json_file:
            data = json.load(json_file)
        split = int(0.8 * len(data))
        d = data
        train_samples = dict(list(d.items())[:split])
        validation_samples = dict(list(d.items())[split:])
        self.reader = MRIReader()
        self.transforms = Compose(
            [ScaleIntensity(self.modality_keys),
             RandomCrop(self.modality_keys, full_vol_dim=self.full_vol_dim, crop_size=self.crop_size),
             DictToNumpy(self.modality_keys)])

        if self.mode == 'train':
            self.list = train_samples

        elif self.mode == 'val':
            self.list = validation_samples

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):

        subject = list(self.list)[index]
        data = {}
        for key in self.modality_keys:
            data[key] = self.reader(self.list[subject][key])
        x, y = self.transforms(data)

        return x, y

# import json
# with open('/mnt/784C5F3A4C5EF1FC/PROJECTS/MedZoo_dev/MedicalZooPytorch/medzoo/datasets/iseg_2017/files/iseg2017train.json', 'r') as json_file:
#     data = json.load(json_file)
# print(data)
# print(len(data))
# print(list(data)[0])
#
# #self.modality_keys = ['T1','T2','label']
#         # print(imgs)
#         # biglist = [list_IDsT1,list_IDsT2,labels]
#         # dict ={}
#         # for i in range(1,11):
#         #     print(i)
#         #     l = sorted(glob.glob(os.path.join(self.training_path, f'subject-{i}-*.img')))
#         #     print(l)
#         #     d = {}
#         #     for item in l:
#         #         subj, subject_id, type = item.split('/')[-1].strip('.img').split('-')
#         #         d[type] = item
#         #         print(subj, subject_id, type)
#         #     dict[f'{subj}{subject_id}'] = d
#         # print(dict)
#         # import json
#         # with open(f'iseg2017{mode}.json', 'w') as fp:
#         #     json.dump(dict, fp)


# self.training_path = self.root + '/iseg_2017/iSeg-2017-Training/'
# self.testing_path = self.root + '/iseg_2017/iSeg-2017-Testing/'
#
# self.list = []
# self.samples = config[self.mode].total_samples
# self.full_volume = None
# self.save_name = self.root + '/iseg_2017/iSeg-2017-Training/iseg2017-list-' + mode + '-samples-' + str(
#     samples) + '.txt'
#
# subvol = '_vol_' + str(crop_dim[0]) + 'x' + str(crop_dim[1]) + 'x' + str(crop_dim[2])
# self.sub_vol_path = self.root + '/iseg_2017/generated/' + mode + subvol + '/'
# print(f'path {self.training_path}')
