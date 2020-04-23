import json
import os
import random
import shutil
import time

import torch
import numpy as np
import torch
import torch.backends.cudnn as cudnn

def reproducibility(args,seed):
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    # FOR FASTER GPU TRAINING WHEN INPUT SIZE DOESN'T VARY
    # LET'S TEST IT
    cudnn.benchmark = True


def save_arguments(args, path):
    with open(path + '/training_arguments.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    f.close()


def datestr():
    now = time.gmtime()
    return '{:02}_{:02}___{:02}_{:02}'.format(now.tm_mday, now.tm_mon, now.tm_hour, now.tm_min)


def shuffle_lists(a, b, seed=777):
    c = list(zip(a, b))
    random.seed(seed)
    random.shuffle(c)
    a, b = zip(*c)
    return a, b


def prepare_input(args, input_tuple):
    if args.inModalities == 4:
        if args.inChannels == 4:
            img_1, img_2, img_3, img_4, target = input_tuple
            input_tensor = torch.cat((img_1, img_2, img_3, img_4), dim=1)
        elif args.inChannels == 3:
            # t1 post constast is ommited
            img_1, _, img_3, img_4, target = input_tuple
            input_tensor = torch.cat((img_1, img_3, img_4), dim=1)
        elif args.inChannels == 2:
            # t1 and t2 only
            img_1, _, img_3, _, target = input_tuple
            input_tensor = torch.cat((img_1, img_3), dim=1)
        elif args.inChannels == 1:
            # t1 only
            input_tensor, _, _, target = input_tuple
    if args.inModalities == 3:
        if args.inChannels == 3:
            img_1, img_2, img_3, target = input_tuple
            input_tensor = torch.cat((img_1, img_2, img_3), dim=1)
        elif args.inChannels == 2:
            img_1, img_2, _, target = input_tuple
            input_tensor = torch.cat((img_1, img_2), dim=1)
        elif args.inChannels == 1:
            input_tensor, _, _, target = input_tuple
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


