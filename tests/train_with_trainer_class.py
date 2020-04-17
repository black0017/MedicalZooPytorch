import argparse
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import lib.medloaders as medical_loaders
import lib.medzoo as medzoo
# Lib files
import lib.utils as utils
from lib.losses3D import DiceLoss
from lib.train.trainer import Trainer


def main():
    args = get_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    ## FOR REPRODUCIBILITY OF RESULTS
    seed = 1777777
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    # FOR FASTER GPU TRAINING WHEN INPUT SIZE DOESN'T VARY
    # cudnn.benchmark = True

    utils.make_dirs(args.save)

    training_generator, val_generator, full_volume, affine = medical_loaders.generate_datasets(args,
                                                                                               path='.././datasets')
    model, optimizer = medzoo.create_model(args)

    criterion = DiceLoss(classes=args.classes)

    if args.cuda:
        model = model.cuda()
        print("Model transferred in GPU.....")

    trainer = Trainer(args, model, criterion, optimizer, train_data_loader=training_generator,
                      valid_data_loader=val_generator, lr_scheduler=None)
    print("START TRAINING...")
    trainer.training()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=1)
    parser.add_argument('--dataset_name', type=str, default="brats2018")
    parser.add_argument('--dim', nargs="+", type=int, default=(32, 32, 32))
    parser.add_argument('--nEpochs', type=int, default=10)
    parser.add_argument('--classes', type=int, default=5)
    parser.add_argument('--samples_train', type=int, default=10)
    parser.add_argument('--samples_val', type=int, default=10)
    parser.add_argument('--inChannels', type=int, default=4)
    parser.add_argument('--inModalities', type=int, default=4)
    parser.add_argument('--split', default=0.8, type=float, help='Select percentage of training data(default: 0.8)')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--model', type=str, default='UNET3D',
                        choices=('VNET', 'VNET2', 'UNET3D', 'DENSENET1', 'DENSENET2', 'DENSENET3', 'HYPERDENSENET'))
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))

    args = parser.parse_args()

    args.save = '../saved_models/' + args.model + '_checkpoints/' + args.model + '_{}_{}_'.format(
        utils.datestr(), args.dataset_name)
    args.tb_log_dir = '../runs/_' + args.model + '_' + args.dataset_name
    return args


if __name__ == '__main__':
    main()
