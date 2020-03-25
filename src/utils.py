import time
import torch
import os
import shutil


def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)


def save_checkpoint(state, is_best, path, prefix, filename='checkpoint.pth.tar'):
    prefix_save = os.path.join(path, prefix)
    name = prefix_save + '_' + filename
    torch.save(state, name)
    if is_best:
        shutil.copyfile(name, prefix_save + '_BEST.pth.tar')


def noop(x):
    return x


def make_dirs(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.makedirs(path)


def create_stats_files(path):
    train_f = open(os.path.join(path, 'train.csv'), 'w')
    val_f = open(os.path.join(path, 'val.csv'), 'w')
    return train_f, val_f


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


def write_train_score(writer, iter, loss_dice, dice_coeff, per_ch_score):
    writer.add_scalar('Train/loss_dice', loss_dice, iter)
    writer.add_scalar('Train/dice_coeff', dice_coeff, iter)
    writer.add_scalar('Train/air', per_ch_score[0], iter)
    writer.add_scalar('Train/csf', per_ch_score[1], iter)
    writer.add_scalar('Train/gm', per_ch_score[2], iter)
    writer.add_scalar('Train/wm', per_ch_score[3], iter)


def write_val_score(writer, test_loss, dice_coeff, avg_air, avg_csf, avg_gm, avg_wm, epoch):
    writer.add_scalar('Val/test_loss', test_loss, epoch)
    writer.add_scalar('Val/dice_coeff', dice_coeff, epoch)
    writer.add_scalar('Val/air', avg_air, epoch)
    writer.add_scalar('Val/csf', avg_csf, epoch)
    writer.add_scalar('Val/gm', avg_gm, epoch)
    writer.add_scalar('Val/wm', avg_wm, epoch)


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


def visualize3d(input_tuple, model, dim, stride=8, in_channels=2):
    """
    this function will produce overlaping windows sub-volume prediction
    that produces full 3d medical image
    compare some slices with ground truth
    :param input_tuple: t1, t2, segment
    :param dim: (d1,d2,d3))
    :param stride: (d1,d2,d3))
    :return: 3d reconstructed volume
    """
    model.eval()

    return None


def visualize_no_overlap(args, input_tuple, model, dim):
    """
    this function will produce NON-overlaping  sub-volumes prediction
    that produces full 3d medical image
    compare some slices with ground truth
    :param input_tuple: t1, t2, segment
    :param dim: (d1,d2,d3))
    :param stride: (d1,d2,d3))
    :return: 3d reconstructed volume
    """
    model.eval()
    t1, t2, segment_map = input_tuple

    B, S, H, W = t1.shape

    model.eval()
    ### CREATE SUB_VOLUMES
    img_t1 = t1.view(-1, dim[0], dim[1], dim[2])
    img_t2 = t2.view(-1, dim[0], dim[1], dim[2])

    predictions = torch.tensor([])
    for i in range(len(img_t1)):
        if args.inChannels == 2:
            input_tensor = torch.cat((img_t1[i].unsqueeze(0).unsqueeze(0), img_t2[i].unsqueeze(0).unsqueeze(0)), dim=1)
        else:
            input_tensor = img_t1[i].unsqueeze(0).unsqueeze(0)

        if args.cuda:
            input_tensor = input_tensor.cuda()
            predictions = predictions.cuda()

        predicted = model(input_tensor)
        predictions = torch.cat((predictions, predicted))

    predictions = predictions.view(-1, 4, S, H, W)
    # 4 classes iSEg
    return  predictions


def inference(model, input):
    """
    subvolume inference
    :param model:
    :param input:
    :return:
    """
    input = input.cuda()
    output = model(input)
    return output.cpu()
