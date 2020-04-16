"""
MICCAI 2019 Medical Deep Learning 2D high resolution image segmentation project:
MICCAI 2019 Prostate Cancer segmentation challenge
Data can be downloaded from here: https://gleason2019.grand-challenge.org/

"""
import argparse
import torch, os

from torch.utils.tensorboard import SummaryWriter

# Lib files
import lib.utils as utils
import lib.medloaders as medical_loaders
import lib.medzoo as medzoo
import lib.train as train

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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

    criterion = medzoo.DiceLoss2D(args.classes)

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
    parser.add_argument('--batchSz', type=int, default=4)
    parser.add_argument('--dataset_name', type=str, default="miccai2019")
    parser.add_argument('--nEpochs', type=int, default=100)
    parser.add_argument('--dim', nargs="+", type=int, default=(256, 256))
    parser.add_argument('--classes', type=int, default=7)
    parser.add_argument('--inChannels', type=int, default=3)
    parser.add_argument('--inModalities', type=int, default=1)
    parser.add_argument('--fold_id', default='1', type=str, help='Select subject for fold validation')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--model', type=str, default='UNET2D',
                        choices=(
                            'VNET', 'VNET2', 'UNET3D', 'DENSENET1', 'DENSENET2', 'DENSENET3', 'HYPERDENSENET',
                            "UNET2D"))
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))

    args = parser.parse_args()

    args.save = '../saved_models/' + args.model + '_checkpoints/' + args.model + '_{}_{}_'.format(
        utils.datestr(), args.dataset_name)
    return args


if __name__ == '__main__':
    main()
