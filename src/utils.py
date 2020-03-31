import time
import torch
import os
import shutil
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import random


def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)


def shuffle_lists(a, b, seed=777):
    c = list(zip(a, b))
    random.seed(seed)
    random.shuffle(c)
    a, b = zip(*c)
    return a, b


def save_checkpoint(state, is_best, path, prefix, filename='checkpoint.pth.tar'):
    prefix_save = os.path.join(path, prefix)
    name = prefix_save + '_' + filename
    torch.save(state, name)
    if is_best:
        shutil.copyfile(name, prefix_save + '_BEST.pth.tar')


def save_model(model, args, dice_loss, epoch, best_pred_loss):
    is_best = False
    if dice_loss < best_pred_loss:
        is_best = True
        best_pred_loss = dice_loss
        save_checkpoint({'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'best_prec1': best_pred_loss},
                        is_best, args.save, args.model + "_best")
    elif epoch % 5 == 0:
        save_checkpoint({'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'best_prec1': best_pred_loss},
                        is_best, args.save, args.model + "_epoch_" + str(epoch))
    return best_pred_loss


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


def visualize_no_overlap(args, full_volume, affine, model, epoch, dim, writer):
    """
    this function will produce NON-overlaping  sub-volumes prediction
    that produces full 3d medical image
    compare some slices with ground truth
    :param full_volume: t1, t2, segment
    :param dim: (d1,d2,d3))
    :return: 3d reconstructed volume
    """
    print(full_volume[0].shape)
    _, slices, height, width = full_volume[0].shape

    ## TODO generalize function - currently in CPU due to memory problems
    args.cuda = False
    classes = args.classes
    model = model.eval()
    if not args.cuda:
        model = model.cpu()

    input_tensor, segment_map = create_3d_subvol(args, full_volume, dim)

    sub_volumes = input_tensor.shape[0]
    predictions = torch.tensor([]).cpu()

    # TODO generalize

    for i in range(sub_volumes):
        predicted = model(input_tensor).cpu()
        predictions = torch.cat((predictions, predicted))

    predictions = predictions.view(-1, classes, slices, height, width).detach()

    save_path_2d_fig = args.save + '/' + 'epoch__' + str(epoch).zfill(4) + '.png'
    create_2d_views(predictions, segment_map, epoch, writer, save_path_2d_fig)

    # TODO test save
    save_path = args.save + '/Pred_volume_epoch_' + str(epoch)
    save_3d_vol(predictions, affine, save_path)


def visualize_offline(args, epoch, model, full_volume, affine, writer, criterion=None):
    model.eval()
    test_loss = 0

    classes, slices, height, width = 4, 144, 192, 256

    predictions = torch.tensor([]).cpu()
    segment_map = torch.tensor([]).cpu()
    for batch_idx, input_tuple in enumerate(full_volume):
        with torch.no_grad():
            t1_path, t2_path, seg_path = input_tuple

            img_t1, img_t2, sub_segment_map = torch.tensor(np.load(t1_path), dtype=torch.float32)[None, None], \
                                              torch.tensor(np.load(t2_path), dtype=torch.float32)[None, None], \
                                              torch.tensor(
                                                  np.load(seg_path), dtype=torch.float32)[None]

            input_tensor, sub_segment_map = prepare_input(args, (img_t1, img_t2, sub_segment_map))
            input_tensor.requires_grad = False

            predicted = model(input_tensor).cpu()
            predictions = torch.cat((predictions, predicted))
            segment_map = torch.cat((segment_map, sub_segment_map.cpu()))

    predictions = predictions.view(-1, classes, slices, height, width).detach()
    segment_map = segment_map.view(-1, slices, height, width).detach()
    save_path_2d_fig = args.save + '/' + 'epoch__' + str(epoch).zfill(4) + '.png'

    create_2d_views(predictions, segment_map, epoch, writer, save_path_2d_fig)

    # TODO test save
    save_path = args.save + '/Pred_volume_epoch_' + str(epoch)
    save_3d_vol(predictions, affine, save_path)

    return test_loss


def create_3d_subvol(args, full_volume, dim):
    if args.inChannels == 3:
        img_1, img_2, img_3, target = full_volume
        print(img_1.shape)

        img_1 = torch.squeeze(img_1, dim=0).view(-1, dim[0], dim[1], dim[2])
        img_2 = img_2.view(-1, dim[0], dim[1], dim[2])
        img_3 = img_3.view(-1, dim[0], dim[1], dim[2])
        input_tensor = torch.stack((img_1, img_2, img_3), dim=1)

    elif args.inChannels == 2:
        img_1, img_2, target = full_volume

        img_1 = img_1.view(-1, dim[0], dim[1], dim[2])
        img_2 = img_2.view(-1, dim[0], dim[1], dim[2])
        input_tensor = torch.stack((img_1, img_2), dim=1)

    elif args.inChannels == 1:
        img_t1, _, target = full_volume
        input_tensor = torch.unsqueeze(img_t1, dim=1)

    return input_tensor, target


def create_2d_views(predictions, segment_map, epoch, writer, path_to_save):
    """
    Comparative 2d vizualization of median slices:
    axial, saggital and transpose. Save to png file and to tensorboard
    :param predictions:
    :param segment_map:
    :param epoch:
    :param writer:
    :param path_to_save:
    :return:
    """
    b, classes, slices, width, height = predictions.shape
    s = int(slices / 2.0)
    h = int(height / 2.0)
    w = int(width / 2.0)
    _, segment_pred = predictions.max(dim=1)
    segment_pred = seg_map_vizualization(segment_pred)

    s1 = segment_pred[0, s, :, :].long()
    s2 = segment_pred[0, :, w, :].long()
    s3 = segment_pred[0, :, :, h].long()

    p1 = segment_map[0, s, :, :].long()
    p2 = segment_map[0, :, w, :].long()
    p3 = segment_map[0, :, :, h].long()

    assert s1.shape == p1.shape
    assert s2.shape == p2.shape
    assert s3.shape == p3.shape

    list_vol = [s1, p1, s2, p2, s3, p3]
    rows, columns = 3, 2
    figure = plt.figure(figsize=(16, 16))
    for i in range(len(list_vol)):
        figure.add_subplot(rows, columns, i + 1)
        plt.imshow(list_vol[i], cmap='gray')

    writer.add_figure('Images/all_2d_views', figure, epoch)
    writer.add_image('Images/pred_view_1', s1, epoch, dataformats='HW')
    writer.add_image('Images/pred_view_2', s2, epoch, dataformats='HW')
    writer.add_image('Images/pred_view_3', s3, epoch, dataformats='HW')

    # TODO save image pairs
    # a1 = torch.stack((s1, p1)).long()
    # a2 = torch.stack((s2, p2)).long()
    # a3 = torch.stack((s3, p3)).long()
    # print(a1.shape,a2.shape,a3.shape)
    # writer.add_images('view_1', a1, epoch, dataformats='NHWC' )
    # writer.add_images('view_2', a2, epoch, dataformats='NHWC' )
    # writer.add_images('view_3', a3, epoch, dataformats='NHWC' )


# Todo  test!
def save_3d_vol(predictions, affine, save_path):
    # np.save(save_path+'.npy', predictions)
    pred_nifti_img = nib.Nifti1Image(predictions, affine)
    nib.save(pred_nifti_img, save_path + '.nii.gz')


def seg_map_vizualization(segmentation_map):
    # visual labels of ISEG-2017
    label_values = [0, 10, 150, 250]
    for c, j in enumerate(label_values):
        segmentation_map[segmentation_map == c] = j
    return segmentation_map


def prepare_input(args, input_tuple):
    if args.inChannels == 3:
        img_1, img_2, img_3, target = input_tuple
        input_tensor = torch.cat((img_1, img_2, img_3), dim=1)
    elif args.inChannels == 2:
        img_t1, img_t2, target = input_tuple
        input_tensor = torch.cat((img_t1, img_t2), dim=1)
    elif args.inChannels == 1:
        img_t1, img_t2, target = input_tuple
        input_tensor = img_t1

    if args.cuda:
        input_tensor, target = input_tensor.cuda(), target.cuda()

    return input_tensor, target


# TODO test!!!!!!
# conf_matrix = tnt.meter.ConfusionMeter(classes)
# conf_matrix.add(y_pred.detach(), y_true)
# plot_confusion_matrix(conf_matrix.conf, list_keys, normalize=False, title="Confusion Matrix TEST - Last epoch")
def plot_confusion_matrix(cm, target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(16, 16))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(title + str(accuracy) + '.png')
    plt.close('all')
    # TODO save to tensorboard!!!!!!!


def expand_as_one_hot(target, classes):
    shape = target.size()
    shape = list(shape)
    shape.insert(1, classes)
    shape = tuple(shape)
    src = target.unsqueeze(1)
    return torch.zeros(shape).to(target.device).scatter_(1, src, 1).squeeze(0)


def add_conf_matrix(target, pred, conf_matrix):
    batch_size = pred.shape[0]
    classes = pred.shape[1]
    target = target.detach().cpu().long()
    target = expand_as_one_hot(target, classes)
    if batch_size == 1:
        tar = target.view(classes, -1).permute(1, 0)
        pr = pred.view(classes, -1).permute(1, 0)
        # Accepts N x K tensors where N is the number of voxels and K the number of classes
        conf_matrix.add(pr, tar)
    else:

        for i in range(batch_size):
            tar = target[i, ...]
            pr = pred[i, ...]
            tar = tar.view(classes, -1).permute(1, 0)
            pr = pr.view(classes, -1).permute(1, 0)
            conf_matrix.add(pr, tar)
    return conf_matrix
