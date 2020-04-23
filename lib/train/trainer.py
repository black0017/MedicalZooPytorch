import numpy as np
import torch

# from utils import inf_loop, MetricTracker
from lib.utils.general import prepare_input
from lib.visual3D_temp.BaseWriter import TensorboardWriter


class Trainer:
    """
    Trainer class
    """

    def __init__(self, args, model, criterion, optimizer, train_data_loader,
                 valid_data_loader=None, lr_scheduler=None):

        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_data_loader = train_data_loader
        # epoch-based training
        self.len_epoch = len(self.train_data_loader)
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(train_data_loader.batch_size))
        self.writer = TensorboardWriter(args)

    def training(self):
        for epoch in range(1, self.args.nEpochs):
            self.train_epoch(epoch)
            if self.do_validation:
                self.validate_epoch(epoch)
            ## TODO WRITER SCALARS END OF EPOCH

            ## TODO SAVE CHECKPOINT
            val_loss = self.writer.data['val']['loss'] / self.writer.data['val']['count']
            if self.args.save != None:
                self.model.save_checkpoint(self.args.save,
                                           epoch, val_loss,
                                           optimizer=self.optimizer,
                                           name=None)
            self.writer._write_end_of_epoch(epoch)

            # RESET writer data
            self.writer.reset('train')
            self.writer.reset('val')

    def train_epoch(self, epoch):
        self.model.train()
        n_processed = 0
        for batch_idx, input_tuple in enumerate(self.train_data_loader):
            self.optimizer.zero_grad()

            input_tensor, target = prepare_input(self.args, input_tuple)
            input_tensor.requires_grad = True
            output = self.model(input_tensor)
            loss_dice, per_ch_score = self.criterion(output, target)
            loss_dice.backward()
            self.optimizer.step()

            partial_epoch = epoch + batch_idx / self.len_epoch - 1

            self.writer.update_scores(batch_idx, loss_dice.item(), per_ch_score, 'train',
                                      epoch * self.len_epoch + batch_idx)
            ## TODO display terminal statistics per batch or iteration steps
            if (batch_idx % 100 == 0):
                self.writer.display_terminal(partial_epoch, epoch, 'train')

        # END OF EPOCH DISPLAY
        self.writer.display_terminal(self.len_epoch, epoch, mode='train', summary=True)

    def validate_epoch(self, epoch):
        self.model.eval()

        for batch_idx, input_tuple in enumerate(self.valid_data_loader):
            with torch.no_grad():
                input_tensor, target = prepare_input(self.args, input_tuple)
                input_tensor.requires_grad = False

                output = self.model(input_tensor)
                loss, per_ch_score = self.criterion(output, target)

                self.writer.update_scores(batch_idx, loss.item(), per_ch_score, 'val',
                                          epoch * self.len_epoch + batch_idx)

        self.writer.display_terminal(len(self.valid_data_loader), epoch, mode='val', summary=True)
