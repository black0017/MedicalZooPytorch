from torch.utils.data import DataLoader
from .iseg2017 import MRIDatasetISEG2017
from .mrbrains2018 import MRIDatasetMRBRAINS2018
from .miccai_2019_pathology import MICCAI2019_gleason_pathology
from .ixi_t1_t2 import IXIMRIdataset
from .brats2018 import MICCAIBraTS2018
from .covid_ct_dataset import CovidCTDataset
from .COVIDxdataset import COVIDxDataset
def generate_datasets(args, path='.././datasets'):
    params = {'batch_size': args.batchSz,
              'shuffle': True,
              'num_workers': 1}
    samples_train = args.samples_train
    samples_val = args.samples_val

    if args.dataset_name == "iseg2017":
        train_loader = MRIDatasetISEG2017('train', dataset_path=path, crop_dim=args.dim,
                                          fold_id=args.fold_id, samples=samples_train, save=True)

        val_loader = MRIDatasetISEG2017('val', dataset_path=path, crop_dim=args.dim, fold_id=args.fold_id,
                                        samples=samples_val, save=True)
    elif args.dataset_name == "mrbrains":
        train_loader = MRIDatasetMRBRAINS2018('train', dataset_path=path, classes=args.classes, dim=args.dim,
                                              fold_id=args.fold_id, samples=samples_train, save=True)

        val_loader = MRIDatasetMRBRAINS2018('val', dataset_path=path, classes=args.classes, dim=args.dim,
                                            fold_id=args.fold_id,
                                            samples=samples_val, save=True)
    elif args.dataset_name == "miccai2019":
        split = (0.8, 0.2)
        total_data = 244
        split_idx = int(split[0] * total_data) - 1

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
        split = (0.8, 0.2)
        total_data = 244
        split_idx = int(split[0] * total_data)
        train_loader = MICCAIBraTS2018('train', dataset_path=path, classes=args.classes, crop_dim=args.dim,
                                       split_idx=split_idx, samples=samples_train, save=True)

        val_loader = MICCAIBraTS2018('val', dataset_path=path, classes=args.classes, crop_dim=args.dim,
                                     split_idx=split_idx,
                                     samples=samples_val, save=True)
    elif args.dataset_name == 'COVID_CT':
        train_loader = CovidCTDataset('train',root_dir='.././datasets/covid_ct_data/',
                                      txt_COVID='.././datasets/covid_ct_data/CT_COVID/trainCT_COVID.txt',
                                      txt_NonCOVID='.././datasets/covid_ct_data/CT_NonCovid/trainCT_NonCOVID.txt')
        val_loader = CovidCTDataset('val',root_dir='.././datasets/covid_ct_dataset',
                                    txt_COVID='.././datasets/covid_ct_data/CT_COVID/valCT_COVID.txt',
                                    txt_NonCOVID='.././datasets/covid_ct_data/CT_NonCovid/valCT_NonCOVID.txt')
    elif args.dataset_name == 'COVIDx':
        train_loader = COVIDxDataset(mode='train', n_classes=args.classes, dataset_path=path,
                                     dim=(224, 224))
        val_loader = COVIDxDataset(mode='test', n_classes=args.classes, dataset_path=path,
                                   dim=(224, 224))
    training_generator = DataLoader(train_loader, **params)
    val_generator = DataLoader(val_loader, **params)





    print("DATA SAMPLES HAVE BEEN GENERATED SUCCESSFULLY")
    return training_generator, val_generator, val_loader.full_volume, val_loader.affine
