# TODO create class Writer
import numpy as np

# inherit
# Dataset voxel labels here
iseg2017 = ["air", "csf", "gm", "wm"]
mrbrains4 = ["air", "csf", "gm", "wm"]
mrbrains8 = ["air", "csf", "gm", "wm"]
brats2018 = ["air", "csf", "gm", "wm"]


def write_score(writer, dataset, iter, loss_dice, per_ch_score, mode="Train/"):
    dice_coeff = np.mean(per_ch_score) * 100

    # choose dataset labels here

    writer.add_scalar(mode + 'loss_dice', loss_dice, iter)
    classes = len(per_ch_score)
    for i in range(classes):
        writer.add_scalar(mode + 'air', per_ch_score[0], iter)
        writer.add_scalar(mode + 'csf', per_ch_score[1], iter)
        writer.add_scalar(mode + 'gm', per_ch_score[2], iter)
        writer.add_scalar(mode + 'wm', per_ch_score[3], iter)


def write_train_val_score(writer, epoch, train_stats, val_stats):
    writer.add_scalars('Loss', {'train': train_stats[0],
                                'val': val_stats[0],
                                }, epoch)
    writer.add_scalars('Coeff', {'train': train_stats[1],
                                 'val': val_stats[1],
                                 }, epoch)

    writer.add_scalars('Air', {'train': train_stats[2],
                               'val': val_stats[2],
                               }, epoch)

    writer.add_scalars('CSF', {'train': train_stats[3],
                               'val': val_stats[3],
                               }, epoch)
    writer.add_scalars('GM', {'train': train_stats[4],
                              'val': val_stats[4],
                              }, epoch)
    writer.add_scalars('WM', {'train': train_stats[5],
                              'val': val_stats[5],
                              }, epoch)
    return

def write_train_val_score(writer, epoch, train_channel_score, val_channel_score):
    assert len(train_channel_score) == len(val_channel_score)
    channels = len(train_channel_score)
    for i in range(channels):
        writer.add_scalars('Loss', {'train': train_channel_score[0],
                                'val': val_channel_score[0],
                                }, epoch)

    return
