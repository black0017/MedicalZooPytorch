import itertools
import matplotlib.pyplot as plt
import numpy as np
import torch


# TODO test!!!!!!
# conf_matrix = tnt.meter.ConfusionMeter(classes)
# conf_matrix.add(y_pred.detach(), y_true)
# plot_confusion_matrix(conf_matrix.conf, list_keys, normalize=False, title="Confusion Matrix TEST - Last epoch")
def plot_confusion_matrix(cm, target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=False):
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


def expand_as_one_hot(target, classes):
    shape = target.size()
    shape = list(shape)
    shape.insert(1, classes)
    shape = tuple(shape)
    src = target.unsqueeze(1).long()
    return torch.zeros(shape).to(target.device).scatter_(1, src, 1).squeeze(0)
