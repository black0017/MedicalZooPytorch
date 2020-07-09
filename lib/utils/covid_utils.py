import torch


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct, len(target), correct / len(target)


def print_stats(args, epoch, num_samples, trainloader, metrics):
    if (num_samples % args.log_interval == 1):
        print("Epoch:{:2d}\tSample:{:5d}/{:5d}\tLoss:{:.4f}\tAccuracy:{:.2f}".format(epoch,
                                                                                     num_samples,
                                                                                     len(
                                                                                         trainloader) * args.batchSz,
                                                                                     metrics.avg('loss')
                                                                                     ,
                                                                                     metrics.avg('accuracy')))


def print_summary(args, epoch, num_samples, metrics, mode=''):
    print(mode + "\n SUMMARY EPOCH:{:2d}\tSample:{:5d}/{:5d}\tLoss:{:.4f}\tAccuracy:{:.2f}\n".format(epoch,
                                                                                                     num_samples,
                                                                                                     num_samples,
                                                                                                     metrics.avg(
                                                                                                         'loss'),
                                                                                                     metrics.avg(
                                                                                                         'accuracy')))


class MetricTracker:
    def __init__(self, *keys, writer=None, mode='/'):

        self.writer = writer
        self.mode = mode + '/'
        self.keys = keys

        self.data = dict.fromkeys(keys, 0)
        self.reset()

    def reset(self):
        for key in self.data:
            self.data[key] = 0

    def update(self, key, value, n=1, writer_step=1):
        if self.writer is not None:
            self.writer.add_scalar(self.mode + '/' + key, value, writer_step)
        self.data[key] += value * n

    def update_all_metrics(self, iteration, values_dict, n=1, writer_step=1):
        self.data['count'] = iteration
        for key in values_dict:
            self.update(key, values_dict[key], n, writer_step)

    def avg_Acc(self, key):
        return self.data['correct'] / self.data['total']

    def print_all_metrics(self):
        s = ''

        for key in self.keys:
            s += "{} {:.4f}\t".format(key, self.data[key] / self.data['count'])

        return s

    def display_terminal(self, iter, epoch, mode='train', summary=False):
        """

        :param iter: iteration or partial epoch
        :param epoch: epoch of training
        :param loss: any loss numpy
        :param mode: train or val ( for training and validation)
        :param summary: to print total statistics at the end of epoch
        """
        if summary:

            info_print = "\n Epoch {:2d} : {} summary".format(epoch, mode)

            for i in range(len(self.keys)):
                info_print += " {} : {:.4f}".format(self.keys[i],
                                                    self.data[self.keys[i]] / self.data['count'])

            print(info_print)
        else:

            info_print = "partial epoch: {:.3f} ".format(iter)

            for i in range(len(self.keys)):
                info_print += " {} : {:.4f}".format(self.keys[i],
                                                    self.data[self.keys[i]] / self.data['count'])
            print(info_print)




def read_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines]
    return txt_data
