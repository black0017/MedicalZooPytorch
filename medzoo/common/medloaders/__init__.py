from torch.utils.data import DataLoader

from .COVIDxdataset import COVIDxDataset
from .Covid_Segmentation_dataset import COVID_Seg_Dataset
from .brats2018 import MICCAIBraTS2018
from .brats2019 import MICCAIBraTS2019
from .brats2020 import MICCAIBraTS2020
from .covid_ct_dataset import CovidCTDataset
from .iseg2017 import MRIDatasetISEG2017
from .iseg2019 import MRIDatasetISEG2019
from .ixi_t1_t2 import IXIMRIdataset
from .miccai_2019_pathology import MICCAI2019_gleason_pathology
from .mrbrains2018 import MRIDatasetMRBRAINS2018


def generate_datasets(args, path='.././datasets'):
    params = {'batch_size': args.batchSz,
              'shuffle': True,
              'num_workers': 2}
    samples_train = args.samples_train
    samples_val = args.samples_val
    split_percent = args.split

    if args.dataset_name == "iseg2017":
        total_data = 10
        split_idx = int(split_percent * total_data)
        train_loader = MRIDatasetISEG2017(args, 'train', dataset_path=path, crop_dim=args.dim,
                                          split_id=split_idx, samples=samples_train, load=args.loadData)

        val_loader = MRIDatasetISEG2017(args, 'val', dataset_path=path, crop_dim=args.dim, split_id=split_idx,
                                        samples=samples_val, load=args.loadData)

    elif args.dataset_name == "iseg2019":
        total_data = 10
        split_idx = int(split_percent * total_data)
        train_loader = MRIDatasetISEG2019(args, 'train', dataset_path=path, crop_dim=args.dim,
                                          split_id=split_idx, samples=samples_train, load=args.loadData)

        val_loader = MRIDatasetISEG2019(args, 'val', dataset_path=path, crop_dim=args.dim, split_id=split_idx,
                                        samples=samples_val, load=args.loadData)
    elif args.dataset_name == "mrbrains4":
        train_loader = MRIDatasetMRBRAINS2018(args, 'train', dataset_path=path, classes=args.classes, dim=args.dim,
                                              split_id=0, samples=samples_train, load=args.loadData)

        val_loader = MRIDatasetMRBRAINS2018(args, 'val', dataset_path=path, classes=args.classes, dim=args.dim,
                                            split_id=0,
                                            samples=samples_val, load=args.loadData)
    elif args.dataset_name == "mrbrains9":
        train_loader = MRIDatasetMRBRAINS2018(args, 'train', dataset_path=path, classes=args.classes, dim=args.dim,
                                              split_id=0, samples=samples_train, load=args.loadData)

        val_loader = MRIDatasetMRBRAINS2018(args, 'val', dataset_path=path, classes=args.classes,
                                            dim=args.dim,
                                            split_id=0,
                                            samples=samples_val, load=args.loadData)
    elif args.dataset_name == "miccai2019":
        total_data = 244
        split_idx = int(split_percent * total_data) - 1

        val_loader = MICCAI2019_gleason_pathology(args, 'val', dataset_path=path, split_idx=split_idx,
                                                  crop_dim=args.dim,
                                                  classes=args.classes, samples=samples_val,
                                                  save=True)

        print('Generating train set...')
        train_loader = MICCAI2019_gleason_pathology(args, 'train', dataset_path=path, split_idx=split_idx,
                                                    crop_dim=args.dim,
                                                    classes=args.classes, samples=samples_train,
                                                    save=True)

    elif args.dataset_name == "ixi":
        loader = IXIMRIdataset(args, dataset_path=path, voxels_space=args.dim, modalities=args.inModalities, save=True)
        generator = DataLoader(loader, **params)
        return generator, loader.affine

    elif args.dataset_name == "brats2018":
        total_data = 244
        split_idx = int(split_percent * total_data)
        train_loader = MICCAIBraTS2018(args, 'train', dataset_path=path, classes=args.classes, crop_dim=args.dim,
                                       split_idx=split_idx, samples=samples_train, load=args.loadData)

        val_loader = MICCAIBraTS2018(args, 'val', dataset_path=path, classes=args.classes, crop_dim=args.dim,
                                     split_idx=split_idx,
                                     samples=samples_val, load=args.loadData)

    elif args.dataset_name == "brats2019":
        split = (0.8, 0.2)
        total_data = 335
        split_idx = int(split[0] * total_data)
        train_loader = MICCAIBraTS2019(args, 'train', dataset_path=path, classes=args.classes, crop_dim=args.dim,
                                       split_idx=split_idx, samples=samples_train, load=args.loadData)

        val_loader = MICCAIBraTS2019(args, 'val', dataset_path=path, classes=args.classes, crop_dim=args.dim,
                                     split_idx=split_idx,
                                     samples=samples_val, load=args.loadData)

    elif args.dataset_name == "brats2020":
        split = (0.8, 0.2)
        total_data = 335
        split_idx = int(split[0] * total_data)
        train_loader = MICCAIBraTS2020(args, 'train', dataset_path=path, classes=args.classes, crop_dim=args.dim,
                                       split_idx=split_idx, samples=samples_train, load=args.loadData)

        val_loader = MICCAIBraTS2020(args, 'val', dataset_path=path, classes=args.classes, crop_dim=args.dim,
                                     split_idx=split_idx,
                                     samples=samples_val, load=args.loadData)
    elif args.dataset_name == 'COVID_CT':
        train_loader = CovidCTDataset('train', root_dir='.././datasets/covid_ct_dataset/',
                                      txt_COVID='.././datasets/covid_ct_dataset/trainCT_COVID.txt',
                                      txt_NonCOVID='.././datasets/covid_ct_dataset/trainCT_NonCOVID.txt')
        val_loader = CovidCTDataset('val', root_dir='.././datasets/covid_ct_dataset',
                                    txt_COVID='.././datasets/covid_ct_dataset/valCT_COVID.txt',
                                    txt_NonCOVID='.././datasets/covid_ct_dataset/valCT_NonCOVID.txt')
    elif args.dataset_name == 'COVIDx':
        train_loader = COVIDxDataset(mode='train', n_classes=args.classes, dataset_path=path,
                                     dim=(224, 224))
        val_loader = COVIDxDataset(mode='val', n_classes=args.classes, dataset_path=path,
                                   dim=(224, 224))

    elif args.dataset_name == 'covid_seg':
        train_loader = COVID_Seg_Dataset(mode='train', dataset_path=path, crop_dim=args.dim,
                                         fold=0, samples=samples_train)

        val_loader = COVID_Seg_Dataset(mode='val', dataset_path=path, crop_dim=args.dim,
                                       fold=0, samples=samples_val)
    training_generator = DataLoader(train_loader, **params)
    val_generator = DataLoader(val_loader, **params)

    print("DATA SAMPLES HAVE BEEN GENERATED SUCCESSFULLY")
    return training_generator, val_generator, val_loader.full_volume, val_loader.affine


def select_full_volume_for_infer(args, path='.././datasets'):
    params = {'batch_size': args.batchSz,
              'shuffle': True,
              'num_workers': 2}
    samples_train = args.samples_train
    samples_val = args.samples_val
    split_percent = args.split

    if args.dataset_name == "iseg2017":
        total_data = 10
        split_idx = int(split_percent * total_data)
        loader = MRIDatasetISEG2017('viz', dataset_path=path, crop_dim=args.dim,
                                    split_id=split_idx, samples=samples_train)


    elif args.dataset_name == "iseg2019":
        total_data = 10
        split_idx = int(split_percent * total_data)
        train_loader = MRIDatasetISEG2019('train', dataset_path=path, crop_dim=args.dim,
                                          split_id=split_idx, samples=samples_train)

        val_loader = MRIDatasetISEG2019('val', dataset_path=path, crop_dim=args.dim, split_id=split_idx,
                                        samples=samples_val)
    elif args.dataset_name == "mrbrains4":
        train_loader = MRIDatasetMRBRAINS2018('train', dataset_path=path, classes=args.classes, dim=args.dim,
                                              split_id=0, samples=samples_train)

        val_loader = MRIDatasetMRBRAINS2018('val', dataset_path=path, classes=args.classes, dim=args.dim,
                                            split_id=0,
                                            samples=samples_val)
    elif args.dataset_name == "mrbrains9":
        train_loader = MRIDatasetMRBRAINS2018('train', dataset_path=path, classes=args.classes, dim=args.dim,
                                              split_id=0, samples=samples_train)

        val_loader = MRIDatasetMRBRAINS2018('val', dataset_path=path, classes=args.classes,
                                            dim=args.dim,
                                            split_id=0,
                                            samples=samples_val)
    elif args.dataset_name == "miccai2019":
        total_data = 244
        split_idx = int(split_percent * total_data) - 1

        val_loader = MICCAI2019_gleason_pathology('val', dataset_path=path, split_idx=split_idx, crop_dim=args.dim,
                                                  classes=args.classes, samples=samples_val,
                                                  save=True)

        print('Generating train set...')
        train_loader = MICCAI2019_gleason_pathology('train', dataset_path=path, split_idx=split_idx, crop_dim=args.dim,
                                                    classes=args.classes, samples=samples_train,
                                                    save=True)

    elif args.dataset_name == "ixi":
        loader = IXIMRIdataset(dataset_path=path, voxels_space=args.dim, modalities=args.inModalities, save=True)
        generator = DataLoader(loader, **params)
        return generator, loader.affine

    elif args.dataset_name == "brats2018":
        total_data = 244
        split_idx = int(split_percent * total_data)
        train_loader = MICCAIBraTS2018('train', dataset_path=path, classes=args.classes, crop_dim=args.dim,
                                       split_idx=split_idx, samples=samples_train)

        val_loader = MICCAIBraTS2018('val', dataset_path=path, classes=args.classes, crop_dim=args.dim,
                                     split_idx=split_idx,
                                     samples=samples_val)

    elif args.dataset_name == "brats2019":
        split = (0.8, 0.2)
        total_data = 335
        split_idx = int(split[0] * total_data)
        train_loader = MICCAIBraTS2018('train', dataset_path=path, classes=args.classes, crop_dim=args.dim,
                                       split_idx=split_idx, samples=samples_train)

        val_loader = MICCAIBraTS2018('val', dataset_path=path, classes=args.classes, crop_dim=args.dim,
                                     split_idx=split_idx,
                                     samples=samples_val)
    elif args.dataset_name == 'COVID_CT':
        train_loader = CovidCTDataset('train', root_dir='.././datasets/covid_ct_dataset/',
                                      txt_COVID='.././datasets/covid_ct_dataset/trainCT_COVID.txt',
                                      txt_NonCOVID='.././datasets/covid_ct_dataset/trainCT_NonCOVID.txt')
        val_loader = CovidCTDataset('val', root_dir='.././datasets/covid_ct_dataset',
                                    txt_COVID='.././datasets/covid_ct_dataset/valCT_COVID.txt',
                                    txt_NonCOVID='.././datasets/covid_ct_dataset/valCT_NonCOVID.txt')
    elif args.dataset_name == 'COVIDx':
        train_loader = COVIDxDataset(mode='train', n_classes=args.classes, dataset_path=path,
                                     dim=(224, 224))
        val_loader = COVIDxDataset(mode='val', n_classes=args.classes, dataset_path=path,
                                   dim=(224, 224))

    elif args.dataset_name == 'covid_seg':
        train_loader = COVID_Seg_Dataset(mode='train', dataset_path=path, crop_dim=args.dim,
                                         fold=0, samples=samples_train)

        val_loader = COVID_Seg_Dataset(mode='val', dataset_path=path, crop_dim=args.dim,
                                       fold=0, samples=samples_val)

    print("DATA SAMPLES HAVE BEEN GENERATED SUCCESSFULLY")
    return loader.full_volume, loader.affine
