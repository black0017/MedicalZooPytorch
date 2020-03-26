from torch.utils.data import DataLoader

from .iseg2017 import MRIDatasetISEG2017
from .mrbrains2018 import MRIDatasetMRBRAINS2018


def generate_datasets(dataset_name="iseg", path='.././datasets/', dim=(32, 32, 32), batch=1, fold_id=1,
                      samples_train=5000,
                      samples_val=1000):
    params = {'batch_size': batch,
              'shuffle': True,
              'num_workers': 1}

    train_loader = MRIDatasetISEG2017('train', dataset_path=path, dim=dim,
                                      fold_id=fold_id, samples=samples_train, save=True)

    val_loader = MRIDatasetISEG2017('val', dataset_path=path, dim=dim, fold_id=fold_id,
                                    samples=samples_val, save=False)

    if len(train_loader) != 0 and len(val_loader) != 0:
        print("data are loaded")
    training_generator = DataLoader(train_loader, **params)
    val_generator = DataLoader(val_loader, **params)

    return training_generator, val_generator, val_loader.full_volume
