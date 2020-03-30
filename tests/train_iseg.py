# Python libraries
import argparse, os
import torch
from torch.utils.tensorboard import SummaryWriter

# Lib files
import src.utils as utils
import src.medloaders as medical_loaders
import src.medzoo as medzoo
import src.train as train

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
seed = 1777777
torch.manual_seed(seed)


def main():
    args = get_arguments()
    utils.make_dirs(args.save)
    train_f, val_f = utils.create_stats_files(args.save)
    name_model = args.model + "_" + args.dataset_name + "_" + utils.datestr()
    writer = SummaryWriter(log_dir='../runs/' + name_model, comment=name_model)

    best_prec1 = 100.
    DIM = (64, 64, 64)
    samples_train = 100
    samples_val = 100

    training_generator, val_generator, full_volume, affine = medical_loaders.generate_datasets(path='.././datasets',
                                                                                               dim=DIM,
                                                                                               batch=args.batchSz,
                                                                                               fold_id=args.fold_id,
                                                                                               samples_train=samples_train,
                                                                                               samples_val=samples_val)
    model, optimizer = medzoo.create_model(args)
    criterion = medzoo.DiceLoss(all_classes=4)

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
        torch.cuda.manual_seed(seed)
        model = model.cuda()
        print("Model transferred in GPU.....")

    print("START TRAINING...")
    for epoch in range(1, args.nEpochs + 1):

        train_stats = train.train_dice(args, epoch, model, training_generator, optimizer, criterion, train_f, writer)

        val_stats = train.test_dice(args, epoch, model, val_generator, criterion, val_f, writer)

        utils.write_train_val_score(writer, epoch, train_stats, val_stats)

        #if epoch % 5 == 0:
            #utils.visualize_no_overlap(args, full_volume, affine, model, epoch, DIM, writer)

        utils.save_model(model, args, val_stats, epoch, best_prec1)

    train_f.close()
    val_f.close()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=4)
    parser.add_argument('--dataset_name', type=str, default="iseg")
    parser.add_argument('--nEpochs', type=int, default=250)
    parser.add_argument('--classes', type=int, default=4)
    parser.add_argument('--inChannels', type=int, default=2)
    parser.add_argument('--inModalities', type=int, default=2)
    parser.add_argument('--fold_id', default='1', type=str, help='Select subject for fold validation')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
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
