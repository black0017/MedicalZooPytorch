from torch.utils.data import DataLoader

from ...datasets.MICCAI_2019_pathology_challenge.loaders.miccai_2019_pathology import MICCAI2019_gleason_pathology
from ...datasets.MICCAI_BraTS_2018_Data_Training.loaders.brats2018 import MICCAIBraTS2018
from ...datasets.MICCAI_BraTS_2018_Data_Training.loaders.brats2019 import MICCAIBraTS2019
from ...datasets.MICCAI_BraTS_2018_Data_Training.loaders.brats2020 import MICCAIBraTS2020
from ...datasets.covid_ct_dataset.loaders.covid_ct_dataset import CovidCTDataset
from ...datasets.covid_x_dataset.loaders.COVIDxdataset import COVIDxDataset
from ...datasets.covid_x_dataset.loaders.Covid_Segmentation_dataset import COVID_Seg_Dataset
from ...datasets.iseg_2017.loaders.iseg2017 import MRIDatasetISEG2017
from ...datasets.iseg_2019.loaders.iseg2019 import MRIDatasetISEG2019
from ...datasets.ixi.loaders.ixi_t1_t2 import IXIMRIdataset
from ...datasets.mrbrains_2018.loaders.mrbrains2018 import MRIDatasetMRBRAINS2018
from ...utils.logger import Logger

LOG = Logger(name='medloader').get_logger()


def generate_datasets(args, path='.././datasets'):
    """

    Args:
        args:
        path:

    Returns:

    """
    params = {'batch_size': args.batchSz,
              'shuffle': True,
              'num_workers': 2}

    if args.dataset_name == "iseg2017":
        train_loader = MRIDatasetISEG2017(args, 'train', dataset_path=path)

        val_loader = MRIDatasetISEG2017(args, 'val', dataset_path=path)

    elif args.dataset_name == "iseg2019":
        train_loader = MRIDatasetISEG2019(args, 'train', dataset_path=path)

        val_loader = MRIDatasetISEG2019(args, 'val', dataset_path=path,)

    elif args.dataset_name == "mrbrains4":
        train_loader = MRIDatasetMRBRAINS2018(args, 'train', dataset_path=path)

        val_loader = MRIDatasetMRBRAINS2018(args, 'val', dataset_path=path)
    elif args.dataset_name == "mrbrains9":
        train_loader = MRIDatasetMRBRAINS2018(args, 'train', dataset_path=path)

        val_loader = MRIDatasetMRBRAINS2018(args, 'val', dataset_path=path,)
    elif args.dataset_name == "miccai2019":
        val_loader = MICCAI2019_gleason_pathology(args, 'val', dataset_path=path)

        LOG.info('Generating train set...')
        train_loader = MICCAI2019_gleason_pathology(args, 'train', dataset_path=path)

    elif args.dataset_name == "ixi":
        loader = IXIMRIdataset(args, dataset_path=path)
        generator = DataLoader(loader, **params)
        return generator, loader.affine

    elif args.dataset_name == "brats2018":
        train_loader = MICCAIBraTS2018(args, 'train', dataset_path=path)

        val_loader = MICCAIBraTS2018(args, 'val', dataset_path=path)

    elif args.dataset_name == "brats2019":

        train_loader = MICCAIBraTS2019(args, 'train', dataset_path=path)

        val_loader = MICCAIBraTS2019(args, 'val', dataset_path=path)

    elif args.dataset_name == "brats2020":
        train_loader = MICCAIBraTS2020(args, 'train', dataset_path=path)

        val_loader = MICCAIBraTS2020(args, 'val', dataset_path=path)
    elif args.dataset_name == 'COVID_CT':
        train_loader = CovidCTDataset('train', root_dir='.././datasets/covid_ct_dataset/',
                                      txt_COVID='.././datasets/covid_ct_dataset/trainCT_COVID.txt',
                                      txt_NonCOVID='.././datasets/covid_ct_dataset/trainCT_NonCOVID.txt')
        val_loader = CovidCTDataset('val', root_dir='.././datasets/covid_ct_dataset',
                                    txt_COVID='.././datasets/covid_ct_dataset/valCT_COVID.txt',
                                    txt_NonCOVID='.././datasets/covid_ct_dataset/valCT_NonCOVID.txt')
    elif args.dataset_name == 'COVIDx':
        train_loader = COVIDxDataset(mode='train', n_classes=args.classes, dataset_path=path)
        val_loader = COVIDxDataset(mode='val', n_classes=args.classes, dataset_path=path)

    elif args.dataset_name == 'covid_seg':
        train_loader = COVID_Seg_Dataset(mode='train', dataset_path=path)

        val_loader = COVID_Seg_Dataset(mode='val', dataset_path=path)

    training_generator = DataLoader(train_loader, **params)
    val_generator = DataLoader(val_loader, **params)

    LOG.info("DATA SAMPLES HAVE BEEN GENERATED SUCCESSFULLY")
    return training_generator, val_generator, val_loader.full_volume, val_loader.affine


def select_full_volume_for_infer(args, path='.././datasets'):
    """

    Args:
        args:
        path:

    Returns:

    """
    params = {'batch_size': args.batchSz,
              'shuffle': True,
              'num_workers': 2}


    if args.dataset_name == "iseg2017":
        loader = MRIDatasetISEG2017('viz', dataset_path=path)

    elif args.dataset_name == "iseg2019":
        train_loader = MRIDatasetISEG2019('train', dataset_path=path)

        val_loader = MRIDatasetISEG2019('val', dataset_path=path)
    elif args.dataset_name == "mrbrains4":
        train_loader = MRIDatasetMRBRAINS2018('train', dataset_path=path)

        val_loader = MRIDatasetMRBRAINS2018('val', dataset_path=path)
    elif args.dataset_name == "mrbrains9":
        train_loader = MRIDatasetMRBRAINS2018('train', dataset_path=path)

        val_loader = MRIDatasetMRBRAINS2018('val', dataset_path=path)
    elif args.dataset_name == "miccai2019":
        val_loader = MICCAI2019_gleason_pathology('val', dataset_path=path)

        print('Generating train set...')
        train_loader = MICCAI2019_gleason_pathology('train', dataset_path=path)

    elif args.dataset_name == "ixi":
        loader = IXIMRIdataset(dataset_path=path, )
        generator = DataLoader(loader, **params)
        return generator, loader.affine

    elif args.dataset_name == "brats2018":
        train_loader = MICCAIBraTS2018('train', dataset_path=path)

        val_loader = MICCAIBraTS2018('val', dataset_path=path)

    elif args.dataset_name == "brats2019":
        train_loader = MICCAIBraTS2018('train', dataset_path=path)

        val_loader = MICCAIBraTS2018('val', dataset_path=path)
    elif args.dataset_name == 'COVID_CT':
        train_loader = CovidCTDataset('train', root_dir='.././datasets/covid_ct_dataset/',
                                      txt_COVID='.././datasets/covid_ct_dataset/trainCT_COVID.txt',
                                      txt_NonCOVID='.././datasets/covid_ct_dataset/trainCT_NonCOVID.txt')
        val_loader = CovidCTDataset('val', root_dir='.././datasets/covid_ct_dataset',
                                    txt_COVID='.././datasets/covid_ct_dataset/valCT_COVID.txt',
                                    txt_NonCOVID='.././datasets/covid_ct_dataset/valCT_NonCOVID.txt')
    elif args.dataset_name == 'COVIDx':
        train_loader = COVIDxDataset(mode='train', n_classes=args.classes, dataset_path=path)
        val_loader = COVIDxDataset(mode='val', n_classes=args.classes, dataset_path=path)

    elif args.dataset_name == 'covid_seg':
        train_loader = COVID_Seg_Dataset(mode='train', dataset_path=path)

        val_loader = COVID_Seg_Dataset(mode='val', dataset_path=path)

    print("DATA SAMPLES HAVE BEEN GENERATED SUCCESSFULLY")
    return loader.full_volume, loader.affine
