import argparse
import os

import lib.medloaders as medical_loaders
import lib.medzoo as medzoo
# Lib files
import lib.utils as utils
from lib.losses3D import DiceLoss, create_loss
from lib.train.trainer import Trainer

import torch
def main():
    args = get_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ## FOR REPRODUCIBILITY OF RESULTS
    seed = 1777777
    utils.reproducibility(args, seed)

    utils.make_dirs(args.save)
    utils.save_arguments(args, args.save)

    training_generator, val_generator, full_volume, affine = medical_loaders.generate_datasets(args,
                                                                                               path='.././datasets')
    model, optimizer = medzoo.create_model(args)
    criterion = create_loss('CrossEntropyLoss')
    criterion = DiceLoss(classes=args.classes, weight=torch.tensor([0.1, 1, 1, 1]).cuda())

    if args.cuda:
        model = model.cuda()
        print("Model transferred in GPU.....")

    trainer = Trainer(args, model, criterion, optimizer, train_data_loader=training_generator,
                      valid_data_loader=val_generator, lr_scheduler=None)
    print("START TRAINING...")
    trainer.training()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=4)
    parser.add_argument('--dataset_name', type=str, default="iseg2017")
    parser.add_argument('--dim', nargs="+", type=int, default=(64, 64, 64))
    parser.add_argument('--nEpochs', type=int, default=250)

    parser.add_argument('--classes', type=int, default=4)
    parser.add_argument('--samples_train', type=int, default=100)
    parser.add_argument('--samples_val', type=int, default=10)
    parser.add_argument('--split', type=float, default=0.8)
    parser.add_argument('--inChannels', type=int, default=4)
    parser.add_argument('--inModalities', type=int, default=4)
    parser.add_argument('--fold_id', default='1', type=str, help='Select subject for fold validation')
    parser.add_argument('--lr', default=5e-3, type=float,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--model', type=str, default='VNET',
                        choices=('VNET', 'VNET2', 'UNET3D', 'DENSENET1', 'DENSENET2', 'DENSENET3', 'HYPERDENSENET'))
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--log_dir', type=str,
                        default='../runs/')
    args = parser.parse_args()

    args.save = '../saved_models/' + args.model + '_checkpoints/' + args.model + '_{}_{}_'.format(
        utils.datestr(), args.dataset_name)
    args.tb_log_dir = '../runs/'
    return args


if __name__ == '__main__':
    main()
