import torch
import lib.utils as utils

"""
Unified train script that keep train/val statistics in Tensorboard
Currently works for 4-class segmentations
"""


def train_dice(args, epoch, model, trainLoader, optimizer, criterion):
    model.train()
    n_processed = 0
    train_loss = 0
    dice_avg_coeff = 0
    avg_air, avg_csf, avg_gm, avg_wm = 0, 0, 0, 0
    stop = 1

    for batch_idx, input_tuple in enumerate(trainLoader):
        optimizer.zero_grad()

        input_tensor, target = utils.prepare_input(args, input_tuple)
        input_tensor.requires_grad = True
        output = model(input_tensor)
        loss_dice, per_ch_score = criterion(output, target)
        loss_dice.backward()
        optimizer.step()

        partial_epoch = epoch + batch_idx / len(trainLoader) - 1
        n_processed = batch_idx + 1
        dice_coeff = 100. * (1. - loss_dice.item())
        avg_air += per_ch_score[0]
        avg_csf += per_ch_score[1]
        avg_gm += per_ch_score[2]
        avg_wm += per_ch_score[3]
        train_loss += loss_dice.item()
        dice_avg_coeff += dice_coeff

        if batch_idx % stop == 0:
            display_status_4_classes(epoch, train_loss, dice_avg_coeff, avg_air, avg_csf, avg_gm, avg_wm,
                                     partial_epoch,
                                     n_processed)
    avg_air = avg_air / n_processed
    avg_csf = avg_csf / n_processed
    avg_gm = avg_gm / n_processed
    avg_wm = avg_wm / n_processed
    dice_avg_coeff = dice_avg_coeff / n_processed
    train_loss = train_loss / n_processed

    display_status_4_classes(epoch, train_loss, dice_avg_coeff, avg_air, avg_csf, avg_gm, avg_wm, summary=True)

    return train_loss, dice_avg_coeff, avg_air, avg_csf, avg_gm, avg_wm


def test_dice(args, epoch, model, testLoader, criterion):
    model.eval()
    test_loss = 0
    avg_dice_coef = 0
    avg_air, avg_csf, avg_gm, avg_wm = 0, 0, 0, 0

    for batch_idx, input_tuple in enumerate(testLoader):
        with torch.no_grad():
            input_tensor, target = utils.prepare_input(args, input_tuple)
            input_tensor.requires_grad = False

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

    display_status_4_classes(epoch, test_loss, coef, avg_air,
                             avg_csf, avg_gm, avg_wm, summary=True)

    return test_loss, coef, avg_air, avg_csf, avg_gm, avg_wm


def display_status_4_classes(epoch, train_loss, dice_avg_coeff, avg_air, avg_csf, avg_gm, avg_wm,
                             partial_epoch=0,
                             n_processed=0, summary=False):
    if not summary:
        print('Train Epoch: {:.2f} \t Dice Loss: {:.4f}\t AVG Dice Coeff: {:.4f} \t'
              'AIR:{:.4f}\tCSF:{:.4f}\tGM:{:.4f}\tWM:{:.4f}\n'.format(
            partial_epoch, train_loss / n_processed, dice_avg_coeff / n_processed, avg_air / n_processed,
                           avg_csf / n_processed, avg_gm / n_processed, avg_wm / n_processed))
    else:
        print(
            '\n\nEpoch Summary: {:.2f} \t Dice Loss: {:.4f}\t AVG Dice Coeff: {:.4f} \t  AIR:{:.4f}\tCSF:{:.4f}\tGM:{:.4f}\tWM:{:.4f}\n\n'.format(
                epoch, train_loss, dice_avg_coeff, avg_air, avg_csf, avg_gm, avg_wm))

