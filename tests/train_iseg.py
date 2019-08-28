#!/usr/bin/env python3
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import shutil

import utils
import medical_loaders
import medical_zoo

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=5)
    parser.add_argument('--dice', action='store_true',default=True)
    parser.add_argument('--nEpochs', type=int, default=400)
    parser.add_argument('--inChannels', type=int, default=1)
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--fold_id', default='1', type=str, help='Select subject for fold validation')

    parser.add_argument('--weight-decay', '--wd', default=1e-8, type=float,
                        metavar='W', help='weight decay (default: 1e-8)')

    parser.add_argument('--lr', default=1e-3, type=float,
                         help='learning rate (default: 1e-3)')

    parser.add_argument('--cuda', action='store_true',default=True)
    parser.add_argument('--save')
    parser.add_argument('--model', type=str, default='VNET2',
                        choices=('VNET','VNET2', 'UNET3D', 'DENSENET1','DENSENET2','DENSENET3','HYPERDENSENET'))
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))

    args = parser.parse_args()
    best_prec1 = 100.
    DIM = (128, 128, 64)

    training_generator, val_generator = generate_datasets(DIM, fold_id=args.fold_id, samples_train=500, samples_val=10)

    args.cuda = args.cuda and torch.cuda.is_available()

    torch.manual_seed(1777777)
    if args.cuda:
        torch.cuda.manual_seed(1777777)

    args.save = args.model + '_checkpoints/' + args.model+'_base_{}_fold_id_{}'.format(utils.datestr(),args.fold_id)
    
    if os.path.exists(args.save):
        shutil.rmtree(args.save)
        os.mkdir(args.save)
    else:
        os.makedirs(args.save)

    weight_decay = args.weight_decay

    print("Building Model . . . . . . . .")

    if args.model == 'VNET2':
        model = medical_zoo.VNetLight(elu=False, nll=False)

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

    print('Number of params: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    if args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.5, weight_decay=weight_decay)
    elif args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(),  lr=args.lr,  weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr,  weight_decay=weight_decay)

    train_f = open(os.path.join(args.save, 'train.csv'), 'w')
    val_f = open(os.path.join(args.save, 'val.csv'), 'w')

    criterion = medical_zoo.DiceLoss()

    print("START TRAINING...")
    for epoch in range(1, args.nEpochs + 1):

        train_dice(args, epoch, model, training_generator, optimizer, criterion, train_f)

        dice_loss = test_dice(args, epoch, model, val_generator, optimizer, criterion, val_f)

        is_best = False
        if dice_loss < best_prec1:
            is_best = True
            best_prec1 = dice_loss

            utils.save_checkpoint({'epoch': epoch,
                             'state_dict': model.state_dict(),
                             'best_prec1': best_prec1},
                            is_best, args.save, args.model + "_best")
        else:
            utils.save_checkpoint({'epoch': epoch,
                             'state_dict': model.state_dict(),
                             'best_prec1': best_prec1},
                            is_best, args.save, args.model+"_last")
    train_f.close()
    val_f.close()


def train_dice(args, epoch, model, trainLoader, optimizer, criterion, trainF):
    model.train()
    n_processed = 0
    n_train = len(trainLoader.dataset)
    train_loss = 0
    dice_avg_coeff = 0
    avg_air, avg_csf, avg_gm, avg_wm = 0,0,0,0
    stop = int(n_train / 20)

    for batch_idx, input_tuple in enumerate(trainLoader):
        optimizer.zero_grad()
        img_t1, img_t2 , target = input_tuple
        if args.inChannels==2:
            input_tensor = torch.cat((img_t1, img_t2),dim=1)
        else:
            input_tensor = img_t1
        input_tensor.requires_grad = True

        if args.cuda:
            input_tensor, target = input_tensor.cuda(), target.cuda()

        output = model(input_tensor)

        loss_dice, per_ch_score = criterion(output, target)
        loss_dice.backward()
        optimizer.step()

        n_processed += 1 # batch size --> TODO

        dice_coeff = 100.*(1. - loss_dice.item())

        avg_air += per_ch_score[0]
        avg_csf += per_ch_score[1]
        avg_gm += per_ch_score[2]
        avg_wm += per_ch_score[3]
        partial_epoch = epoch + batch_idx / len(trainLoader) - 1

        train_loss += loss_dice.item()
        dice_avg_coeff += dice_coeff



        if(batch_idx%stop==0):
            print('Train Epoch: {:.2f} [{}/{}] \t Dice Loss: {:.4f}\t AVG Dice Coeff: {:.4f} \t'
                  'AIR:{:.4f}\tCSF:{:.4f}\tGM:{:.4f}\tWM:{:.4f}\n'.format(
                partial_epoch, n_processed, n_train,
                train_loss/n_processed, dice_avg_coeff/n_processed, avg_air/n_processed,
                avg_csf/n_processed, avg_gm/n_processed, avg_wm/n_processed))


        if(batch_idx==(n_train-1)):
            print('\nEpoch Summary: {:.2f} [{}/{}] \t Dice Loss: {:.4f}\t AVG Dice Coeff: {:.4f}'.format(
                partial_epoch, n_processed, n_train,
                train_loss/n_processed, dice_avg_coeff/n_processed, avg_air/n_processed,
                avg_csf/n_processed, avg_gm/n_processed, avg_wm/n_processed))


        trainF.write('{},{},{}\n'.format(partial_epoch, loss_dice.item(), dice_avg_coeff/n_train ))
        trainF.flush()


def test_dice(args, epoch, model, testLoader, optimizer, criterion, testF):
    model.eval()
    test_loss = 0
    avg_dice_coef = 0
    avg_air, avg_csf, avg_gm, avg_wm = 0,0,0,0

    for batch_idx, input_tuple in enumerate(testLoader):
        img_t1, img_t2, target = input_tuple

        if args.cuda:
            img_t1, target = img_t1.cuda(), target.cuda()

        output = model(img_t1)
        loss , per_ch_score = criterion(output, target)
        test_loss += loss.item()
        avg_dice_coef += (1. - loss.item())

        avg_air += per_ch_score[0]
        avg_csf += per_ch_score[1]
        avg_gm += per_ch_score[2]
        avg_wm += per_ch_score[3]

    nTotal = len(testLoader)
    test_loss /= nTotal
    coef = 100.*avg_dice_coef/nTotal
    avg_air = avg_air/nTotal
    avg_csf = avg_csf/nTotal
    avg_gm = avg_gm/nTotal
    avg_wm = avg_wm/nTotal

    print('\n\n\nTest set: {} \t Dice Loss: {:.4f}\t AVG Dice Coeff: {:.4f} \t'
          'AIR:{:.4f}\tCSF:{:.4f}\tGM:{:.4f}\tWM:{:.4f}\n\n\n'.format(
        epoch, test_loss, coef, avg_air,
        avg_csf, avg_gm, avg_wm))

    testF.write('{},{},{},{},{},{},{}\n'.format(epoch, test_loss,coef, avg_air, avg_csf,avg_gm,avg_wm))
    testF.flush()
    return test_loss

def adjust_opt(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        if epoch < 150:
            lr = 1e-1
        elif epoch == 150:
            lr = 1e-2
        elif epoch == 225:
            lr = 1e-3
        else:
            return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def generate_datasets(dim, fold_id=1, samples_train=500, samples_val=100):
    params = {'batch_size': 1,
              'shuffle': True,
              'num_workers': 1}

    train_loader = medical_loaders.MRIDatasetISEG2017('train', dataset_path='./data', dim=dim,
                                                       fold_id=fold_id, samples=samples_train)

    val_loader =medical_loaders.MRIDatasetISEG2017('val', dataset_path='./data', dim=dim, fold_id=fold_id,
                                                     samples=samples_val)

    training_generator = DataLoader(train_loader, **params)
    val_generator = DataLoader(val_loader, **params)
    print("DATA SAMPLES HAVE BEEN GENERATED SUCCESFULLY")
    return training_generator, val_generator


if __name__ == '__main__':
    main()
