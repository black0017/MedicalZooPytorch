import os
import shutil

import torch

"""
Not used anymore. Will be removed soon.
They now exist in the base class of the models.
"""
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
                        is_best, args.save, args.model + "_best_"+ str(epoch))
    elif epoch % 5 == 0:
        save_checkpoint({'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'best_prec1': best_pred_loss},
                        is_best, args.save, args.model + "_epoch_" + str(epoch))
    return best_pred_loss


def load_checkpoint(checkpoint_path, model):
    if not os.path.exists(checkpoint_path):
        raise IOError("Checkpoint '{checkpoint_path}' does not exist")

    state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state['state_dict'])
    epoch = state['epoch']
    best_pred = state['best_prec1']
    print("=> loaded checkpoint epoch {}"
          .format(state['epoch']))

    return model, epoch, best_pred