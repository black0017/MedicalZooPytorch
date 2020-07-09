# Python libraries
import argparse, os
import torch

# Lib files
import lib.utils as utils
import lib.medloaders as medical_loaders
import lib.medzoo as medzoo
import lib.train as train
from lib.losses3D import DiceLoss

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed = 1777777
torch.manual_seed(seed)


def main():
    args = get_arguments()
    utils.reproducibility(args, seed)
    utils.make_dirs(args.save)
    print("DATASET {} CPKT {}".format(args.dataset_name, args.save))
    training_generator, val_generator, test_generator, val_volume, test_volume, affine = medical_loaders.generate_datasets(
        args,
        path='.././datasets')
    model, optimizer = medzoo.create_model(args)

    criterion = DiceLoss(classes=11, skip_index_after=args.classes)

    if args.cuda:
        model = model.cuda()

    trainer = train.Trainer(args, model, criterion, optimizer, train_data_loader=training_generator,
                            valid_data_loader=val_generator, lr_scheduler=None, val_volume=val_volume,
                            test_volume=test_volume)
    trainer.training()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=4)
    parser.add_argument('--dataset_name', type=str, default="mrbrains4")
    parser.add_argument('--dim', nargs="+", type=int, default=(64, 64, 48))
    parser.add_argument('--nEpochs', type=int, default=100)
    parser.add_argument('--inChannels', type=int, default=3)
    parser.add_argument('--inModalities', type=int, default=3)
    parser.add_argument('--samples_train', type=int, default=1024)
    parser.add_argument('--samples_val', type=int, default=128)
    parser.add_argument('--classes', type=int, default=4)
    parser.add_argument('--terminal_show_freq', default=50000)
    parser.add_argument('--threshold', default=0.01, type=float)
    parser.add_argument('--augmentation', action='store_true', default=True)
    parser.add_argument('--normalization', default='global_mean', type=str,
                        help='Tensor normalization: options max, mean, global')

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--split', default=0.9, type=float, help='Select percentage of training data(default: 0.8)')

    parser.add_argument('--lr', default=1e-2, type=float,
                        help='learning rate (default: 1e-3)')

    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--loadData', default=False)
    parser.add_argument('--model', type=str, default='UNET3D',
                        choices=('VNET', 'VNET2', 'UNET3D', 'DENSENET1', 'DENSENET2', 'DENSENET3', 'HYPERDENSENET'))
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--log_dir', type=str,
                        default='../runs/')

    args = parser.parse_args()

    args.save = '../saved_models/' + args.model + '_checkpoints/' + args.model + '_{}_{}_'.format(
        utils.datestr(), args.dataset_name)
    return args


if __name__ == '__main__':
    main()
