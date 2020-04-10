import shutil
import torch
import os
import random, time


def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)


def shuffle_lists(a, b, seed=777):
    c = list(zip(a, b))
    random.seed(seed)
    random.shuffle(c)
    a, b = zip(*c)
    return a, b


def prepare_input(args, input_tuple):
    if args.inModalities == 3:
        if args.inChannels == 3:
            img_1, img_2, img_3, target = input_tuple
            input_tensor = torch.cat((img_1, img_2, img_3), dim=1)
        elif args.inChannels == 2:
            img_1, img_2, _, target = input_tuple
            input_tensor = torch.cat((img_1, img_2), dim=1)
        elif args.inChannels == 1:
            input_tensor, _, _,target = input_tuple
    elif args.inModalities == 2:
        if args.inChannels == 2:
            img_t1, img_t2, target = input_tuple
            input_tensor = torch.cat((img_t1, img_t2), dim=1)
        elif args.inChannels == 1:
            input_tensor, _, target = input_tuple
    elif args.inModalities == 1:
        input_tensor, target = input_tuple

    if args.cuda:
        input_tensor, target = input_tensor.cuda(), target.cuda()

    return input_tensor, target


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



# Reconstruction + KL divergence losses summed over all elements and batch
def loss_vae(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE, KLD
