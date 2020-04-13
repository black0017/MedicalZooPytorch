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


# TODO train and evaluate will be generalized and added to lib/train.py
def train_old(args, epoch, model, trainLoader, optimizer, criterion, trainF):
    model.train()
    n_train = len(trainLoader.dataset)
    train_loss = 0
    dice_avg_coeff = 0

    for batch_idx, input_tuple in enumerate(trainLoader):
        optimizer.zero_grad()
        input_tensor, target = input_tuple
        input_tensor.requires_grad = True
        input_tensor, target = input_tensor.cuda(), target.cuda()
        output = model(input_tensor)
        loss_dice, per_ch_score = criterion(output.squeeze(0), target)
        loss_dice.backward()
        optimizer.step()

        # Keep Training statistics
        dice_coeff = 100. * (1. - loss_dice.item())
        train_loss += loss_dice.item()
        dice_avg_coeff += dice_coeff

    # Mean statistics
    train_loss = train_loss / n_train
    dice_avg_coeff = dice_avg_coeff / n_train
    print('\n Train Epoch {} Summary:  \t Dice Loss: {:.4f}\t AVG Dice Coeff: {:.4f}'
          .format(epoch, train_loss, dice_avg_coeff))

    trainF.write('{},{},{}\n'.format(epoch, train_loss, dice_avg_coeff))
    trainF.flush()


def evaluate_old(args, epoch, model, val_generator, criterion, val_f):
    import torchnet as tnt
    conf_matrix = tnt.meter.ConfusionMeter(7)
    list_keys = ["c1", "c2", "c3", "c4", "c5", "c6", "c7"]

    model.eval()
    n_val = len(val_generator.dataset)
    val_loss = 0
    dice_avg_coeff = 0
    with torch.no_grad():
        for batch_idx, input_tuple in enumerate(val_generator):
            input_tensor, target = input_tuple
            input_tensor, target = input_tensor.cuda(), target.cuda()
            output = model(input_tensor)

            # loss_dice, per_ch_score = criterion(output.squeeze(0), target)
            loss_dice, per_ch_score = criterion(output, target)
            # Keep Training statistics
            dice_coeff = 100. * (1. - loss_dice.item())
            val_loss += loss_dice.item()
            dice_avg_coeff += dice_coeff

            conf_matrix = utils.add_conf_matrix(target.detach().cpu(), output.detach().cpu(), conf_matrix)
    title_name = args.save + "/Confusion Matrix Epoch_" + str(epoch) + '_score_'
    utils.plot_confusion_matrix(conf_matrix.conf, list_keys, normalize=False,
                                title=title_name)

    # Mean statistics
    val_loss = val_loss / n_val
    dice_avg_coeff = dice_avg_coeff / n_val
    print('\n Validation Epoch {} Summary:  \t Dice Loss: {:.4f}\t AVG Dice Coeff: {:.4f}'.format(epoch, val_loss,
                                                                                                  dice_avg_coeff))
    val_f.write('{},{},{}\n'.format(epoch, val_loss, dice_avg_coeff))
    val_f.flush()
    return val_loss


def main():
    args = get_arguments()
    utils.make_dirs(args.save)
    train_f, val_f = utils.create_stats_files(args.save)

    name_model = args.model + "_" + args.dataset_name + "_" + utils.datestr()
    writer = SummaryWriter(log_dir='../runs/' + name_model, comment=name_model)

    best_pred = 1.01
    samples_train = 200
    samples_val = 200
    training_generator, val_generator, full_volume, affine = medical_loaders.generate_datasets(args,
                                                                                               path='.././datasets',
                                                                                               samples_train=samples_train,
                                                                                               samples_val=samples_val)

    model, optimizer = medzoo.create_model(args)
    criterion = medzoo.DiceLoss2D(args.classes)

    if args.cuda:
        torch.cuda.manual_seed(seed)
        model = model.cuda()

    for epoch in range(1, args.nEpochs + 1):
        train_stats = train.train_dice(args, epoch, model, training_generator, optimizer, criterion, train_f, writer)
        val_stats = train.test_dice(args, epoch, model, val_generator, criterion, val_f, writer)

        utils.write_train_val_score(writer, epoch, train_stats, val_stats)
        best_pred = utils.save_model(model=model, args=args, dice_loss=val_stats[0], epoch=epoch,
                                     best_pred_loss=best_pred)

    train_f.close()
    val_f.close()


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
