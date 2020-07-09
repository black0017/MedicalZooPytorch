# Python libraries
import argparse
import lib.medzoo as medzoo


def main():
    args = get_arguments()
    summary_flag = True

    model_list = ["RESNET3DVAE", 'UNET3D', 'DENSENET1',
                  'DENSENET2', 'DENSENET3', 'HYPERDENSENET', "SKIPDENSENET3D",
                  "DENSEVOXELNET", 'VNET', 'VNET2']

    if summary_flag:
        for key in model_list:
            args.model = key
            model, _ = medzoo.create_model(args)
            print('created...',key)
            # model.test()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=1)
    parser.add_argument('--dataset_name', type=str, default="iseg")
    parser.add_argument('--dim', nargs="+", type=int, default=(16, 16, 16))
    parser.add_argument('--nEpochs', type=int, default=250)
    parser.add_argument('--classes', type=int, default=4)
    parser.add_argument('--inChannels', type=int, default=2)
    parser.add_argument('--inModalities', type=int, default=2)
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--model', type=str, default='UNET3D',
                        choices=('VNET', 'VNET2', 'UNET3D', 'DENSENET1', 'DENSENET2', 'DENSENET3', 'HYPERDENSENET'))
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
