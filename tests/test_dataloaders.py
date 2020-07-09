# Python libraries
import argparse

# Lib files
import lib.utils as utils
import lib.medloaders as medical_loaders


class TestDataLoaders:
    def __init__(self, batch=1, dim=64, classes=10):
        self.batch = batch
        self.dim = dim
        self.classes = classes
        self.binary_classes = 2
        self.args = self.get_arguments()

    def MRBRAINS_4_class(self):
        self.args.dataset_name = "mrbrains"
        training_generator, val_generator, full_volume, affine = medical_loaders.generate_datasets(self.args,
                                                                                                   path='.././datasets')
        print("mrbrains 4 OK!", len(training_generator), len(val_generator))

    def MRBRAINS_9_class(self):
        self.args.classes = 9
        training_generator, val_generator, full_volume, affine = medical_loaders.generate_datasets(self.args,
                                                                                                   path='.././datasets')
        print("mrbrains 8 OK!", len(training_generator), len(val_generator))

    def ISEG2017(self):
        self.args.inChannels = 2
        self.args.inModalities = 2
        self.args.dataset_name = "iseg2017"
        training_generator, val_generator, full_volume, affine = medical_loaders.generate_datasets(self.args,
                                                                                                   path='.././datasets')

        print("iseg  OK! ", len(training_generator), len(val_generator))

    def brats2018(self):
        self.args.inChannels = 4
        self.args.inModalities = 4
        self.args.classes = 5
        self.args.dataset_name = "brats2018"
        training_generator, val_generator, full_volume, affine = medical_loaders.generate_datasets(self.args,
                                                                                                   path='.././datasets')
        print("brats2018  OK!", len(training_generator), len(val_generator))

    def miccai2019(self):
        self.args.dim = (64, 64)
        self.args.inChannels = 3
        self.args.inModalities = 1
        self.args.classes = 7
        self.args.dataset_name = "miccai2019"
        training_generator, val_generator, full_volume, affine = medical_loaders.generate_datasets(self.args,
                                                                                                   path='.././datasets')
        print("miccai2019  OK!", len(training_generator), len(val_generator))

    def ixi(self):
        self.args.inChannels = 2
        self.args.inModalities = 2
        self.args.dim = (1, 1, 1)
        self.args.dataset_name = "ixi"
        generator, affine = medical_loaders.generate_datasets(self.args, path='.././datasets')
        print("ixi  OK!", len(generator))

    def get_arguments(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--batchSz', type=int, default=1)
        parser.add_argument('--dataset_name', type=str, default="mrbrains")
        parser.add_argument('--dim', nargs="+", type=int, default=(16, 16, 16))
        parser.add_argument('--nEpochs', type=int, default=300)
        parser.add_argument('--inChannels', type=int, default=3)
        parser.add_argument('--inModalities', type=int, default=3)
        parser.add_argument('--samples_train', type=int, default=10)
        parser.add_argument('--samples_val', type=int, default=10)
        parser.add_argument('--classes', type=int, default=4)
        parser.add_argument('--fold_id', default='1', type=str, help='Select subject for fold validation')

        parser.add_argument('--lr', default=1e-3, type=float,
                            help='learning rate (default: 1e-3)')

        parser.add_argument('--cuda', action='store_true', default=False)

        parser.add_argument('--model', type=str, default='UNET3D',
                            choices=('VNET', 'VNET2', 'UNET3D', 'DENSENET1', 'DENSENET2', 'DENSENET3', 'HYPERDENSENET'))
        parser.add_argument('--opt', type=str, default='sgd',
                            choices=('sgd', 'adam', 'rmsprop'))

        args = parser.parse_args()

        return args


test_obj = TestDataLoaders(batch=1, dim=64, classes=10)

test_obj.MRBRAINS_4_class()
test_obj.MRBRAINS_9_class()
test_obj.ISEG2017()
test_obj.brats2018()
test_obj.miccai2019()
# test_obj.ixi()
