# Python libraries
import argparse, os
import torch
from torch.utils.tensorboard import SummaryWriter

# Lib files
import lib.utils as utils
import lib.medloaders as medical_loaders
import lib.medzoo as medzoo

import lib.train as train
from lib.losses3D.dice import DiceLoss

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed = 1777777
torch.manual_seed(seed)


def main():
    args = get_arguments()
    utils.make_dirs(args.save)
    name_model = args.model + "_" + args.dataset_name + "_" + utils.datestr()

    # TODO visual3D_temp.Basewriter package
    writer = SummaryWriter(log_dir='../runs/' + name_model, comment=name_model)

    training_generator, val_generator, full_volume, affine = medical_loaders.generate_datasets(args,
                                                                                               path='.././datasets')
    model, optimizer = medzoo.create_model(args)

    criterion = DiceLoss(classes=11, skip_index_after=args.classes)

    if args.cuda:
        torch.cuda.manual_seed(seed)
        model = model.cuda()
        print("Model transferred in GPU.....")

    print("START TRAINING...")
    for epoch in range(1, args.nEpochs + 1):
        train_stats = train.train_dice(args, epoch, model, training_generator, optimizer, criterion)

        val_stats = train.test_dice(args, epoch, model, val_generator, criterion)

        #old
        utils.write_train_val_score(writer, epoch, train_stats, val_stats)

        model.save_checkpoint(args.save, epoch, val_stats[0], optimizer=optimizer)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=8)
    parser.add_argument('--dataset_name', type=str, default="mrbrains")
    parser.add_argument('--dim', nargs="+", type=int, default=(16, 16, 16))
    parser.add_argument('--classes', type=int, default=9)
    parser.add_argument('--nEpochs', type=int, default=10)
    parser.add_argument('--inChannels', type=int, default=3)
    parser.add_argument('--inModalities', type=int, default=3)
    parser.add_argument('--samples_train', type=int, default=10)
    parser.add_argument('--samples_val', type=int, default=10)

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--fold_id', default='1', type=str, help='Select subject for fold validation')

    parser.add_argument('--lr', default=1e-4, type=float,
                        help='learning rate (default: 1e-3)')

    parser.add_argument('--cuda', action='store_true', default=False)

    parser.add_argument('--model', type=str, default='UNET3D',
                        choices=('VNET', 'VNET2', 'UNET3D', 'DENSENET1', 'DENSENET2', 'DENSENET3', 'HYPERDENSENET'))
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))

    args = parser.parse_args()

    args.save = '../saved_models/' + args.model + '_checkpoints/' + args.model + '_{}_{}_'.format(
        utils.datestr(), args.dataset_name)
    return args


if __name__ == '__main__':
    main()
