import os

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from lib.utils import datestr

"""
Under construction ......
"""
dict_class_names = {"iseg2017": ["air", "csf", "gm", "wm"],
                    "mrbrains4": ["air", "csf", "gm", "wm"],
                    "mrbrains8": ["air", "csf", "gm", "wm", "class5", "class6", "class7", "class8"],
                    "brats2018": ["c1", "c2", "c3", "c4", "c5"],
                    "covid_seg":["c1", "c2", "c3"]}


# TODO remove tensorboard x dependency make it work just with tensorboard
class TensorboardWriter():

    def __init__(self, args):
        name_model = args.model + "_" + args.dataset_name + "_" + datestr()
        self.writer = SummaryWriter(log_dir=args.tb_log_dir, comment=name_model)

        # self.step = 0
        # self.mode = ''
        self.csv_train, self.csv_val = self.create_stats_files(args.save)
        self.dataset_name = args.dataset_name
        self.label_names = dict_class_names[args.dataset_name]
        self.data = {"train": dict((label, 0.0) for label in self.label_names),
                     "val": dict((label, 0.0) for label in self.label_names)}
        self.data['train']['loss'] = 0.0
        self.data['val']['loss'] = 0.0
        self.data['train']['count'] = 1.0
        self.data['val']['count'] = 1.0

        self.data['train']['dsc'] = 0.0
        self.data['val']['dsc'] = 0.0

        # self.tb_writer_ftns = {
        #     'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
        #     'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding'
        # }
        #
        # self.timer = datetime.now()

    def display_terminal(self, iter, epoch, mode='train', summary=False):
        """

        :param iter: iteration or partial epoch
        :param epoch: epoch of training
        :param loss: any loss numpy
        :param mode: train or val ( for training and validation)
        :param summary: to print total statistics at the end of epoch
        """
        if summary:

            info_print = "\n Epoch {:2d} : {} summary Loss : {:.4f} DSC : {:.4f}  ".format(epoch, mode,
                                                                                           self.data[mode]['loss'] /
                                                                                           self.data[mode]['count'],
                                                                                           self.data[mode]['dsc'] /
                                                                                           self.data[mode]['count'])

            for i in range(len(self.label_names)):
                info_print += " {} : {:.4f}".format(self.label_names[i],
                                                    self.data[mode][self.label_names[i]] / self.data[mode]['count'])

            print(info_print)
        else:

            info_print = "partial epoch: {:.3f} Loss : {:.4f} DSC : {:.4f}".format(iter, self.data[mode]['loss'] /
                                                                                   self.data[mode]['count'],
                                                                                   self.data[mode]['dsc'] /
                                                                                   self.data[mode]['count'])

            for i in range(len(self.label_names)):
                info_print += " {} : {:.4f}".format(self.label_names[i],
                                                    self.data[mode][self.label_names[i]] / self.data[mode]['count'])
            print(info_print)

    def create_stats_files(self, path):
        train_f = open(os.path.join(path, 'train.csv'), 'w')
        val_f = open(os.path.join(path, 'val.csv'), 'w')
        return train_f, val_f

    def reset(self, mode):
        self.data[mode]['dsc'] = 0.0
        self.data[mode]['loss'] = 0.0
        self.data[mode]['count'] = 1
        for i in range(len(self.label_names)):
            self.data[mode][self.label_names[i]] = 0.0

    def update_scores(self, iter, loss, channel_score, mode, writer_step):
        """

        :param iter: iteration or partial epoch
        :param loss: any loss torch.tensor.item()
        :param channel_score: per channel score or dice coef
        :param mode: train or val ( for training and validation)
        :param writer_step: tensorboard writer step
        """
        ## WARNING ASSUMING THAT CHANNELS IN SAME ORDER AS DICTIONARY  ###########

        dice_coeff = np.mean(channel_score) * 100

        num_channels = len(channel_score)
        self.data[mode]['dsc'] += dice_coeff
        self.data[mode]['loss'] += loss
        self.data[mode]['count'] = iter + 1

        for i in range(num_channels):
            self.data[mode][self.label_names[i]] += channel_score[i]
            if self.writer != None:
                self.writer.add_scalar(mode + '/' + self.label_names[i], channel_score[i], global_step=writer_step)

    def _write_end_of_epoch(self, epoch):

        self.writer.add_scalars('DSC/', {'train': self.data['train']['dsc'] / self.data['train']['count'],
                                         'val': self.data['val']['dsc'] / self.data['val']['count'],
                                         }, epoch)
        self.writer.add_scalars('Loss/', {'train': self.data['train']['loss'] / self.data['train']['count'],
                                          'val': self.data['val']['loss'] / self.data['val']['count'],
                                          }, epoch)
        for i in range(len(self.label_names)):
            self.writer.add_scalars(self.label_names[i],
                                    {'train': self.data['train'][self.label_names[i]] / self.data['train']['count'],
                                     'val': self.data['val'][self.label_names[i]] / self.data['train']['count'],
                                     }, epoch)

        #    TODO write labels accuracies in csv files

        train_csv_line = 'Epoch:{:2d} Loss:{:.4f} DSC:{:.4f}'.format(epoch,
                                                                     self.data['train']['loss'] / self.data['train'][
                                                                         'count'],
                                                                     self.data['train']['dsc'] / self.data['train'][
                                                                         'count'])
        val_csv_line = 'Epoch:{:2d} Loss:{:.4f} DSC:{:.4f}'.format(epoch,
                                                                   self.data['val']['loss'] / self.data['val'][
                                                                       'count'],
                                                                   self.data['val']['dsc'] / self.data['val'][
                                                                       'count'])
        self.csv_train.write(train_csv_line + '\n')
        self.csv_val.write(val_csv_line + '\n')
