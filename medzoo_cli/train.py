# Python libraries
import argparse
import os

import medzoo.common.medloaders as medical_loaders
import medzoo.train as medzoo_trainer
import medzoo.utils as utils
from medzoo.common.losses3D import DiceLoss
from medzoo.models.create_model import create_model
from medzoo.utils.config_reader import ConfigReader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed = 1777777


def train():
    """

    """
    args = get_arguments()

    config = ConfigReader.read_config(args.model_config,args.dataset_config,  args.model, args.dataset)
    config.model = args.model
    config.dataset = args.dataset

    config.save = 'saved_models/' + args.model + '_checkpoints/' + args.model + '_{}_{}_'.format(
        utils.datestr(), args.dataset)

    utils.reproducibility(config, seed)
    utils.make_dirs(config.save)

    training_generator, val_generator, full_volume, affine = medical_loaders.generate_datasets(config,
                                                                                               path='medzoo/datasets')
    model, optimizer = create_model(config)
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
    parser.add_argument('--dataset', type=str, default="iseg2017")
    parser.add_argument('--model', type=str, default='UNET3D',
                        choices=(
                        'VNET', 'VNET2', 'UNET3D', 'DENSENET1', 'DENSENET2', 'DENSENET3', 'HYPERDENSENET'))
    parser.add_argument('--dataset_config', type=str)
    parser.add_argument('--model_config', type=str)


    args = parser.parse_args()

    return args


if __name__ == "__main__":
    train()
