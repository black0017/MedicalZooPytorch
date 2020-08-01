# Python libraries
import argparse
import os

import medzoo.common.medloaders as medical_loaders
import medzoo.models as medzoo
import medzoo.train as medzoo_trainer
import medzoo.utils as utils
from medzoo.common.losses3D import DiceLoss
from medzoo.utils.config_reader import ConfigReader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed = 1777777


def train():
    """

    """
    args = get_arguments()

    config = ConfigReader.read_config(args.config)
    config.model = args.model
    config.dataset_name = args.dataset_name

    config.save = 'saved_models/' + args.model + '_checkpoints/' + args.model + '_{}_{}_'.format(
        utils.datestr(), args.dataset_name)

    utils.reproducibility(config, seed)
    utils.make_dirs(config.save)

    training_generator, val_generator, full_volume, affine = medical_loaders.generate_datasets(config,
                                                                                               path='medzoo/datasets')
    model, optimizer = medzoo.create_model(config)
    criterion = DiceLoss(classes=config.classes)

    if config.cuda:
        model = model.cuda()

    trainer = medzoo_trainer.Trainer(config, model, criterion, optimizer, train_data_loader=training_generator,
                            valid_data_loader=val_generator, lr_scheduler=None)
    trainer.training()


def get_arguments():
    """

    Returns:

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="iseg2019")
    parser.add_argument('--model', type=str, default='UNET3D',
                        choices=('VNET', 'VNET2', 'UNET3D', 'DENSENET1', 'DENSENET2', 'DENSENET3', 'HYPERDENSENET'))
    parser.add_argument('--config', type=str, default='medzoo/defaults.yaml')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    train()
