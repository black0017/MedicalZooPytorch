from torch.utils.data import DataLoader

from .iseg2017 import MRIDatasetISEG2017
from .mrbrains2018 import MRIDatasetMRBRAINS2018


def generate_datasets(dataset_name="iseg", path='.././datasets/', classes=4, dim=(32, 32, 32), batch=1, fold_id=1,
                      samples_train=5000,
                      samples_val=1000):
    params = {'batch_size': batch,
              'shuffle': True,
              'num_workers': 1}

    if dataset_name == "iseg":
        train_loader = MRIDatasetISEG2017('train', dataset_path=path, dim=dim,
                                          fold_id=fold_id, samples=samples_train, save=True)

        val_loader = MRIDatasetISEG2017('val', dataset_path=path, dim=dim, fold_id=fold_id,
                                        samples=samples_val, save=True)
    elif dataset_name == "mrbrains":

        train_loader = MRIDatasetMRBRAINS2018('train', dataset_path=path, classes=classes, dim=dim,
                                              fold_id=fold_id, samples=samples_train, save=True)

        val_loader = MRIDatasetMRBRAINS2018('val', dataset_path=path, classes=classes, dim=dim, fold_id=fold_id,
                                            samples=samples_val, save=True)

    training_generator = DataLoader(train_loader, **params)
    val_generator = DataLoader(val_loader, **params)

    print("DATA SAMPLES HAVE BEEN GENERATED SUCCESFULLY")
    return training_generator, val_generator, val_loader.full_volume, val_loader.affine
