from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from lib.utils import datestr

"""
Under construction ......
"""
dict_class_names = {"iseg2017": ["air", "csf", "gm", "wm"],
                    "mrbrains4": ["air", "csf", "gm", "wm"],
                    "mrbrains8": ["air", "csf", "gm", "wm", "class5", "class6", "class7", "class8"],
                    "brats2018": ["c1", "c2", "c3", "c4", "c5"]}


# TODO remove tensorboard x dependency make it work just with tensorboard
class TensorboardWriter():

    def __init__(self, args):
        name_model = args.model + "_" + args.dataset_name + "_" + datestr()
        self.writer = SummaryWriter(log_dir=args.log_dir + name_model, comment=name_model)

        self.step = 0
        self.mode = ''
        self.csv_train, self.csv_val = self.create_stats_files(args.save)
        self.dataset_name = args.dataset_name
        self.label_names = dict_class_names[args.dataset_name]

        # Average training stats per epoch
        self.train_scores_mean = None
        self.val_scores_mean = None
        self.epoch_samples = 0
        self.mean_train_loss = 0
        self.mean_val_los = 0

        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding'
        }

        self.timer = datetime.now()

    def display_terminal(self, iter, loss, per_channels_score=None, mode='Train', summary=False):
        """

        :param iter: epoch or partial epoch
        :param loss: any loss numpy
        :param per_channels_score:
        :param mode:
        :param summary:
        :return:
        """
        if summary:
            print("\n epoch", iter, ':', mode, 'summary', 'Loss:', loss)
            if per_channels_score is not None:
                for i in range(len(per_channels_score)):
                    print(self.label_names[i], ":", per_channels_score[i])
        else:
            print("partial epoch:", iter, 'Loss:', loss)
            for i in range(len(per_channels_score)):
                print(self.label_names[i], ":", per_channels_score[i])

        # TODO write csv files
        # self.csv_val.write

        return

    def create_stats_files(self, path):
        train_f = open(os.path.join(path, 'train.csv'), 'w')
        val_f = open(os.path.join(path, 'val.csv'), 'w')
        return train_f, val_f

    def write_train_val_score(self, epoch, loss_train, loss_val, train_channel_score, val_channel_score):
        assert len(train_channel_score) == len(val_channel_score)
        dice_coeff_tr = np.mean(train_channel_score) * 100
        dice_coeff_val = np.mean(val_channel_score) * 100
        channels = len(train_channel_score)
        self.writer.add_scalars('DSC/', {'train': dice_coeff_tr,
                                         'val': dice_coeff_val,
                                         }, epoch)
        self.writer.add_scalars('Loss/', {'train': loss_train,
                                          'val': loss_val,
                                          }, epoch)
        for i in range(channels):
            self.writer.add_scalars(self.label_names[i], {'train': train_channel_score[i],
                                                          'val': val_channel_score[i],
                                                          }, epoch)

        if self.epoch_samples == 0:
            self.train_scores_mean = train_channel_score
            self.val_scores_mean = val_channel_score
            self.epoch_samples+=1
        else:
            self.train_scores_mean = (self.train_scores_mean * self.epoch_samples + train_channel_score)/(self.epoch_samples+1)
            self.val_scores_mean = (self.val_scores_mean * self.epoch_samples + val_channel_score)/(self.epoch_samples+1)
            self.epoch_samples = self.epoch_samples+1

