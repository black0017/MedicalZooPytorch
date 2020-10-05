import os

import numpy as np
from torch.utils.tensorboard import SummaryWriter

import medzoo.utils as utils
from medzoo.utils.logger import Logger

LOG = Logger(name='trainer').get_logger()


dict_class_names = {"iseg2017": ["Air", "CSF", "GM", "WM"],
                    "iseg2019": ["Air", "CSF", "GM", "WM"],
                    "mrbrains4": ["Air", "CSF", "GM", "WM"],
                    "mrbrains9": ["Background", "Cort.GM", "BS", "WM", "WML", "CSF",
                                  "Ventr.", "Cerebellum", "stem"],
                    "brats2018": ["Background", "NCR/NET", "ED", "ET"],
                    "brats2019": ["Background", "NCR", "ED", "NET", "ET"],
                    "brats2020": ["Background", "NCR/NET", "ED", "ET"],
                    "covid_seg": ["c1", "c2", "c3"],
                    "miccai2019": ["c1", "c2", "c3", "c4", "c5", "c6", "c7"]
                    }


class TensorboardWriter():
    """
    Concentrate all the training progress of the model in this class.
    It uses tensorboard writer.
    """
    def __init__(self, args):

        name_model = args.model_config.log_dir + args.model_config.model + "_" + args.dataset_config.dataset_name + "_" + utils.datestr()
        self.writer = SummaryWriter(log_dir=args.model_config.log_dir + name_model, comment=name_model)

        utils.make_dirs(args.save)
        self.csv_train, self.csv_val = self.create_stats_files(args.save)
        self.dataset_name = args.dataset_config.dataset_name
        self.classes = args.dataset_config.classes
        self.label_names = dict_class_names[args.dataset_config.dataset_name]

        self.data = self.create_data_structure()

    def create_data_structure(self):
        """
        Initializes the data that we ll be keeping track of.
        Returns: dictionary that contains all the train/val data
        """
        data = {"train": dict((label, 0.0) for label in self.label_names),
                "val": dict((label, 0.0) for label in self.label_names)}
        data['train']['loss'] = 0.0
        data['val']['loss'] = 0.0
        data['train']['count'] = 1.0
        data['val']['count'] = 1.0
        data['train']['dsc'] = 0.0
        data['val']['dsc'] = 0.0
        return data

    def display_terminal(self, iter, epoch, mode='train', summary=False):
        """
        Method to display train stats on terminal

        Args:
            iter: iteration or partial epoch
            epoch: epoch of training
            loss: any loss numpy
            mode: train or val ( for training and validation)
            summary: to print total statistics at the end of epoch
        """        
       
        if summary:
            info_print = "\nSummary {} Epoch {:2d}:  Loss:{:.4f} \t DSC:{:.4f}  ".format(mode, epoch,
                                                                                         self.data[mode]['loss'] /
                                                                                         self.data[mode]['count'],
                                                                                         self.data[mode]['dsc'] /
                                                                                         self.data[mode]['count'])

            for i in range(len(self.label_names)):
                info_print += "\t{} : {:.4f}".format(self.label_names[i],
                                                     self.data[mode][self.label_names[i]] / self.data[mode]['count'])

            LOG.info(info_print)
        else:

            info_print = "\nEpoch: {:.2f} Loss:{:.4f} \t DSC:{:.4f}".format(iter, self.data[mode]['loss'] /
                                                                            self.data[mode]['count'],
                                                                            self.data[mode]['dsc'] /
                                                                            self.data[mode]['count'])

            for i in range(len(self.label_names)):
                info_print += "\t{}:{:.4f}".format(self.label_names[i],
                                                   self.data[mode][self.label_names[i]] / self.data[mode]['count'])
            LOG.info(info_print)

    def create_stats_files(self, path):
        """
        Creates csv files to save the train stats so you can process them out of tensorboard

        Args:
            path: where to create the data

        Returns: train and val path

        """
        train_f = open(os.path.join(path, 'train.csv'), 'w')
        val_f = open(os.path.join(path, 'val.csv'), 'w')
        return train_f, val_f

    def reset(self, mode):
        """
        Resets train statistics for every epoch

        Args:
            mode:
        """
        self.data[mode]['dsc'] = 0.0
        self.data[mode]['loss'] = 0.0
        self.data[mode]['count'] = 1
        for i in range(len(self.label_names)):
            self.data[mode][self.label_names[i]] = 0.0

    def update_scores(self, iter, loss, channel_score, mode, writer_step):
        """[summary]

        Args:
            iter: iteration or partial epoch
            loss: any loss torch.tensor.item()
            channel_score: per channel score or dice coef
            mode: train or val ( for training and validation)
            writer_step: tensorboard writer step
        """        
       
        # TODO WARNING ASSUMING THAT CHANNELS IN SAME ORDER AS DICTIONARY

        dice_coeff = np.mean(channel_score) * 100

        num_channels = len(channel_score)
        self.data[mode]['dsc'] += dice_coeff
        self.data[mode]['loss'] += loss
        self.data[mode]['count'] = iter + 1

        for i in range(num_channels):
            self.data[mode][self.label_names[i]] += channel_score[i]
            if self.writer is not None:
                self.writer.add_scalar(mode + '/' + self.label_names[i], channel_score[i], global_step=writer_step)

    def write_end_of_epoch(self, epoch):
        """
        Updates training stats at the end of epoch + tensorboard

        Args:
            epoch: the current epoch
        """
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
