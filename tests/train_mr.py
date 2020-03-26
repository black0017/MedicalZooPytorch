#!/usr/bin/env python3
import argparse
import torch

import os
import src.utils as utils
import src.medloaders as medical_loaders
import src.medzoo as medical_zoo
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=3)
    parser.add_argument('--dataset_name', type=str, default="mrbrains")
    parser.add_argument('--nEpochs', type=int, default=100)
    parser.add_argument('--inChannels', type=int, default=3)
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
    DIM = (32, 32, 32)
    samples_train = 50
    samples_val = 10
    seed = 1777777
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)

    training_generator, val_generator = medical_loaders.generate_datasets(path='.././datasets', dim=DIM,
                                                                          batch=args.batchSz, fold_id=args.fold_id,
                                                                          samples_train=samples_train,
                                                                          samples_val=samples_val)
    model, optimizer = medical_zoo.create_model(args.model, args.opt, args.lr, args.inChannels)
    criterion = medical_zoo.DiceLoss()

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

        train_stats = train_dice(args, epoch, model, training_generator, optimizer, criterion, train_f, writer)

        val_stats = test_dice(args, epoch, model, val_generator, optimizer, criterion, val_f, writer)

        utils.write_train_val_score(writer, epoch, train_stats, val_stats)

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


def train_dice(args, epoch, model, trainLoader, optimizer, criterion, trainF, writer):
    model.train()
    n_processed = 0
    n_train = len(trainLoader)
    train_loss = 0
    dice_avg_coeff = 0
    avg_air, avg_csf, avg_gm, avg_wm = 0, 0, 0, 0
    stop = int(n_train / 4)

    for batch_idx, input_tuple in enumerate(trainLoader):
        optimizer.zero_grad()
        img_t1, img_t2, target = input_tuple
        batch_size = img_t1.shape[0]
        if args.inChannels == 2:
            input_tensor = torch.cat((img_t1, img_t2), dim=1)
        else:
            input_tensor = img_t1
        input_tensor.requires_grad = True

        if args.cuda:
            input_tensor, target = input_tensor.cuda(), target.cuda()

        output = model(input_tensor)
        loss_dice, per_ch_score = criterion(output, target)
        loss_dice.backward()
        optimizer.step()

        n_processed += batch_size
        dice_coeff = 100. * (1. - loss_dice.item())
        avg_air += per_ch_score[0]
        avg_csf += per_ch_score[1]
        avg_gm += per_ch_score[2]
        avg_wm += per_ch_score[3]
        train_loss += loss_dice.item()
        dice_avg_coeff += dice_coeff
        iter = epoch * n_train + batch_idx

        utils.write_train_score(writer, iter, loss_dice, dice_coeff, per_ch_score)

        if batch_idx % stop == 0:
            partial_epoch = epoch + batch_idx / len(trainLoader) - 1
            print('Train Epoch: {:.2f} [{}/{}] \t Dice Loss: {:.4f}\t AVG Dice Coeff: {:.4f} \t'
                  'AIR:{:.4f}\tCSF:{:.4f}\tGM:{:.4f}\tWM:{:.4f}\n'.format(
                partial_epoch, n_processed, n_train,
                train_loss / n_processed, dice_avg_coeff / n_processed, avg_air / n_processed,
                avg_csf / n_processed, avg_gm / n_processed, avg_wm / n_processed))

    avg_air = avg_air / n_processed
    avg_csf = avg_csf / n_processed
    avg_gm = avg_gm / n_processed
    avg_wm = avg_wm / n_processed
    dice_avg_coeff = dice_avg_coeff / n_processed
    train_loss = train_loss / n_processed

    print(
        '\nEpoch Summary: {:.2f} \t Dice Loss: {:.4f}\t AVG Dice Coeff: {:.4f} \t  AIR:{:.4f}\tCSF:{:.4f}\tGM:{:.4f}\tWM:{:.4f}'.format(
            epoch, train_loss, dice_avg_coeff, avg_air, avg_csf, avg_gm, avg_wm))

    trainF.write('{},{},{},{},{},{},{}\n'.format(epoch, train_loss, dice_avg_coeff, avg_air, avg_csf, avg_gm, avg_wm))
    trainF.flush()

    return train_loss, dice_avg_coeff, avg_air, avg_csf, avg_gm, avg_wm


def test_dice(args, epoch, model, testLoader, optimizer, criterion, testF, writer):
    model.eval()
    test_loss = 0
    avg_dice_coef = 0
    avg_air, avg_csf, avg_gm, avg_wm = 0, 0, 0, 0

    for batch_idx, input_tuple in enumerate(testLoader):
        img_t1, img_t2, target = input_tuple
        if args.inChannels == 2:
            input_tensor = torch.cat((img_t1, img_t2), dim=1)
        else:
            input_tensor = img_t1

        if args.cuda:
            input_tensor, target = input_tensor.cuda(), target.cuda()

        output = model(input_tensor)
        loss, per_ch_score = criterion(output, target)
        test_loss += loss.item()
        avg_dice_coef += (1. - loss.item())

        avg_air += per_ch_score[0]
        avg_csf += per_ch_score[1]
        avg_gm += per_ch_score[2]
        avg_wm += per_ch_score[3]


    nTotal = len(testLoader)
    test_loss /= nTotal
    coef = 100. * avg_dice_coef / nTotal
    avg_air = avg_air / nTotal
    avg_csf = avg_csf / nTotal
    avg_gm = avg_gm / nTotal
    avg_wm = avg_wm / nTotal

    print('\n\n\nTest set: {} \t Dice Loss: {:.4f}\t AVG Dice Coeff: {:.4f} \t'
          'AIR:{:.4f}\tCSF:{:.4f}\tGM:{:.4f}\tWM:{:.4f}\n\n\n'.format(
        epoch, test_loss, coef, avg_air,
        avg_csf, avg_gm, avg_wm))

    utils.write_val_score(writer, test_loss, coef, avg_air, avg_csf, avg_gm, avg_wm, epoch)
    testF.write('{},{},{},{},{},{},{}\n'.format(epoch, test_loss, coef, avg_air, avg_csf, avg_gm, avg_wm))
    testF.flush()

    #utils.visualize3d(testLoader.full_volume, model, DIM, stride=8)


    return test_loss, coef, avg_air, avg_csf, avg_gm, avg_wm





if __name__ == '__main__':
    main()
