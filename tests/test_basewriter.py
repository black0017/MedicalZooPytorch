import argparse

import numpy as np

from lib.utils import datestr
from lib.visual3D_temp import TensorboardWriter


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=1)
    parser.add_argument('--dataset_name', type=str, default="iseg2017")
    parser.add_argument('--dim', nargs="+", type=int, default=(32, 32, 32))
    parser.add_argument('--nEpochs', type=int, default=250)
    parser.add_argument('--classes', type=int, default=4)
    parser.add_argument('--samples_train', type=int, default=10)
    parser.add_argument('--samples_val', type=int, default=10)
    parser.add_argument('--inChannels', type=int, default=2)
    parser.add_argument('--inModalities', type=int, default=2)
    parser.add_argument('--fold_id', default='1', type=str, help='Select subject for fold validation')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--model', type=str, default='DENSEVOXELNET',
                        choices=('VNET', 'VNET2', 'UNET3D', 'DENSENET1', 'DENSENET2', 'DENSENET3', 'HYPERDENSENET'))
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--log_dir', type=str,
                        default='../runs/')

    args = parser.parse_args()

    args.save = '../saved_models/' + args.model + '_checkpoints/' + args.model + '_{}_{}_'.format(
        datestr(), args.dataset_name)
    return args


def TEST_the_writer():
    args = get_arguments()
    test_writer = TensorboardWriter(args)
    for epoch in range(10):
        for i in range(1, 100):
            test_writer.update_scores(iter=i, loss=np.random.rand(1)[0], channel_score=np.random.rand(4), mode='train',
                                      writer_step=epoch * 100 + i)
            # print(test_writer.data['train'])
            # test_writer.display_terminal(i, mode='train')
        test_writer.display_terminal(i, epoch, mode='train', summary=True)
        for i in range(100):
            test_writer.update_scores(iter=i, loss=np.random.rand(1)[0], channel_score=np.random.rand(4), mode='val',
                                      writer_step=epoch * 100 + i)
        test_writer.display_terminal(i, epoch, mode='val', summary=True)
        test_writer.write_end_of_epoch(epoch=epoch)

        # print(test_writer.data)
        test_writer.reset('train')
        test_writer.reset('val')


TEST_the_writer()

print("test complete")
