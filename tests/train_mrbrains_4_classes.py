#!/usr/bin/env python3
import argparse
import torch

import os
import src.utils as utils
import src.medloaders as medical_loaders
import src.medzoo as medical_zoo
from torch.utils.tensorboard import SummaryWriter

import src.train as train

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=4)
    parser.add_argument('--dataset_name', type=str, default="mrbrains")
    parser.add_argument('--dice', action='store_true', default=True)
    parser.add_argument('--nEpochs', type=int, default=300)
    parser.add_argument('--inChannels', type=int, default=3)
    parser.add_argument('--inModalities', type=int, default=3)
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--fold_id', default='1', type=str, help='Select subject for fold validation')

    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate (default: 1e-3)')

    parser.add_argument('--cuda', action='store_true', default=True)

    parser.add_argument('--save')
    parser.add_argument('--model', type=str, default='UNET3D',
                        choices=('VNET', 'VNET2', 'UNET3D', 'DENSENET1', 'DENSENET2', 'DENSENET3', 'HYPERDENSENET'))
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))

    args = parser.parse_args()
    args.save = '../saved_models/' + args.model + '_checkpoints/' + args.model + '_base_{}_fold_id_{}'.format(
        utils.datestr(), args.fold_id)
    utils.make_dirs(args.save)
    train_f, val_f = utils.create_stats_files(args.save)
    name_model = args.model + utils.datestr()
    writer = SummaryWriter(log_dir='../runs/' + name_model, comment=name_model)

    best_prec1 = 100.
    DIM = (128, 128, 32)
    samples_train = 1500
    samples_val = 300
    seed = 1777777
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)

    training_generator, val_generator, full_volume, affine = medical_loaders.generate_datasets(
        dataset_name=args.dataset_name, path='.././datasets', dim=DIM,
        batch=args.batchSz,
        fold_id=args.fold_id,
        samples_train=samples_train,
        samples_val=samples_val)
    model, optimizer = medical_zoo.create_model(args.model, args.opt, args.lr, args.inChannels)
    criterion = medical_zoo.DiceLoss(all_classes=8, desired_classes=4)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.cuda:
        model = model.cuda()
        print("Model transferred in GPU.....")

    print("START TRAINING...")
    for epoch in range(1, args.nEpochs + 1):

        train_stats = train.train_dice(args, epoch, model, training_generator, optimizer, criterion, train_f, writer)

        val_stats = train.test_dice(args, epoch, model, val_generator, criterion, val_f, writer)

        utils.write_train_val_score(writer, epoch, train_stats, val_stats)

        # TODO
        #if epoch % 5 == 0:
            #utils.visualize_no_overlap(args, full_volume, model, epoch, DIM, writer)

        dice_loss = val_stats[0]
        is_best = False
        if dice_loss < best_prec1:
            is_best = True
            best_prec1 = dice_loss

            utils.save_checkpoint({'epoch': epoch,
                                   'state_dict': model.state_dict(),
                                   'best_prec1': best_prec1},
                                  is_best, args.save, args.model + "_best")
        elif epoch % 10 == 0:
            utils.save_checkpoint({'epoch': epoch,
                                   'state_dict': model.state_dict(),
                                   'best_prec1': best_prec1},
                                  is_best, args.save, args.model + "_last")
    train_f.close()
    val_f.close()


if __name__ == '__main__':
    main()
