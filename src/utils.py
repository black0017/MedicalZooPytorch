import time
import torch
import os
import shutil
import matplotlib.pyplot as plt


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


def visualize_no_overlap(args, input_tuple, model, epoch, dim, writer, classes=4):
    """
    this function will produce NON-overlaping  sub-volumes prediction
    that produces full 3d medical image
    compare some slices with ground truth
    :param input_tuple: t1, t2, segment
    :param dim: (d1,d2,d3))
    :return: 3d reconstructed volume
    """
    model.eval()
    t1, t2, segment_map = input_tuple

    B, S, H, W = t1.shape
    ### CREATE SUB_VOLUMES
    img_t1 = t1.view(-1, dim[0], dim[1], dim[2])
    img_t2 = t2.view(-1, dim[0], dim[1], dim[2])

    sub_volumes = len(img_t1)

    predictions = torch.tensor([]).cpu()

    for i in range(sub_volumes):
        if args.inChannels == 2:
            input_tensor = torch.cat((img_t1[i].unsqueeze(0).unsqueeze(0), img_t2[i].unsqueeze(0).unsqueeze(0)), dim=1)
        else:
            input_tensor = img_t1[i].unsqueeze(0).unsqueeze(0)

        if args.cuda:
            input_tensor = input_tensor.cuda()

        predicted = model(input_tensor).cpu()
        predictions = torch.cat((predictions, predicted))

    predictions = predictions.view(-1, classes, S, H, W).detach()
    path_2d_fig = args.save + '/' + 'epoch__' + str(epoch).zfill(4) + '.png'
    create_2d_views(predictions, segment_map, epoch, writer, path_2d_fig)

    #save_path = args.save + '/Pred_volume_epoch_' + str(epoch)
    #save_3d_vol(predictions, affinne, path):


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
    b, classes, slices, height, width = predictions.shape
    s = int(slices / 2.0)
    h = int(height / 2.0)
    w = int(width / 2.0)
    _, segment_pred = predictions.max(dim=1)
    segment_pred = seg_map_vizualization(segment_pred)

    s1 = segment_pred[0, s, :, :].long()
    s2 = segment_pred[0, :, h, :].long()
    s3 = segment_pred[0, :, :, w].long()

    p1 = segment_map[s, :, :].long()
    p2 = segment_map[:, h, :].long()
    p3 = segment_map[:, :, w].long()

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


# Todo save as 3d medical images. requires affine matrix!!!!!
def save_3d_vol(predictions, affinne, path):
    # np.save(path+'.npy', predictions)
    # predictions = nib.Nifti1Image(predictions, np.eye(4))
    # nib.save(predictions, save_path+'.nii.gz')
    # print("DONE")
    return 0


def seg_map_vizualization(segmentation_map):
    # visual labels of ISEG-2017
    label_values = [0, 10, 150, 250]
    for c, j in enumerate(label_values):
        segmentation_map[segmentation_map == c] = j
    return segmentation_map
