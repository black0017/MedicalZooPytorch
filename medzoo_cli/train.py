# Python libraries
import argparse
import os

import medzoo.common.medloaders as medical_loaders
import medzoo.models as medzoo
import medzoo.train as train
# Lib files
import medzoo.utils as utils
from medzoo.common.losses3D import DiceLoss
from medzoo.utils.config_reader import ConfigReader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed = 1777777


def train():
    args = get_arguments()

    config = ConfigReader.read_config(args.config)
    utils.reproducibility(config.training, seed)
    utils.make_dirs(args.save)

    training_generator, val_generator, full_volume, affine = medical_loaders.generate_datasets(config.training,
                                                                                               path='../medzoo/datasets')
    model, optimizer = medzoo.create_model(config.training)
    criterion = DiceLoss(classes=config.training.classes)

    if config.training.cuda:
        model = model.cuda()

    trainer = train.Trainer(config.training, model, criterion, optimizer, train_data_loader=training_generator,
                            valid_data_loader=val_generator)
    trainer.training()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="brats2019")
    parser.add_argument('--model', type=str, default='UNET3D',
                        choices=('VNET', 'VNET2', 'UNET3D', 'DENSENET1', 'DENSENET2', 'DENSENET3', 'HYPERDENSENET'))
    parser.add_argument('--config', type=str, default='../medzoo/defaults.yaml')

    args = parser.parse_args()

    args.save = '../saved_models/' + args.model + '_checkpoints/' + args.model + '_{}_{}_'.format(
        utils.datestr(), args.dataset_name)
    return args


if __name__ == "__main__":
    train()
