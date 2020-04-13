# TODO write tests for all loss functions
import torch

# these losses work with target 4D shape [batch, dim_1, dim_2, dim_3 ]
from lib.losses3D.BCE_dice import BCEDiceLoss
from lib.losses3D.generalized_dice import GeneralizedDiceLoss
from lib.losses3D.dice import DiceLoss
from lib.losses3D.weight_smooth_l1 import WeightedSmoothL1Loss
from lib.losses3D.tags_angular_loss import TagsAngularLoss
from lib.losses3D.ContrastiveLoss import ContrastiveLoss
from lib.losses3D.weight_cross_entropy import WeightedCrossEntropyLoss

# not working yet - todo
from lib.losses3D.pixel_wise_cross_entropy import PixelWiseCrossEntropyLoss


class TestCriterion:
    def __init__(self, batch=1, dim=64, classes=10):
        self.batch = batch
        self.dim = dim
        self.classes = classes
        self.binary_classes = 2

        self.batch_shape_target = (batch, dim, dim, dim)
        self.batch_shape_input = (batch, classes, dim, dim, dim)
        torch.manual_seed(9646456451)

        self.predicted = torch.rand(self.batch_shape_input, requires_grad=True)
        self.target = torch.rand(self.batch_shape_target).long()

    def test_BCEDiceLoss(self):
        classes = 2
        predicted = torch.rand(self.batch, classes, self.dim, self.dim, self.dim)
        target = torch.rand(self.batch, self.dim, self.dim, self.dim)
        criterion = BCEDiceLoss(classes=classes)
        loss, scores = criterion(predicted, target)
        assert (loss.item() > 0) and (loss.item() < 3)
        print("BCEDiceLoss is ok!", loss.item())

    def test_DiceLoss(self):
        criterion = DiceLoss(classes=self.classes)
        loss, per_channel = criterion(self.predicted, self.target)
        assert (loss.item() > 0) and (loss.item() < 1)
        print("DiceLoss is ok!", loss.item())

    def test_generalized_diceLoss(self):
        criterion = GeneralizedDiceLoss(classes=self.classes)
        loss, per_channel = criterion(self.predicted, self.target)
        assert (loss.item() > 0) and (loss.item() < 2)
        print("GeneralizedDiceLoss is ok!", loss.item())

    def test_WeightedCrossEntropyLoss(self):
        criterion = WeightedCrossEntropyLoss()
        loss = criterion(self.predicted, self.target)
        assert (loss.item() > 0) and (loss.item() < 10)
        print("WeightedCrossEntropyLoss is ok!", loss.item())

    def test_ContrastiveLoss(self):
        criterion = ContrastiveLoss()
        loss = criterion(self.predicted, self.target)
        assert (loss.item() > 0) and (loss.item() < 1)
        print("ContrastiveLoss is ok!", loss.item())

    def test_PixelWiseCrossEntropyLoss(self):
        # TODO does not work!!!!!
        print("error PixelWiseCrossEntropyLoss")
        weights = torch.rand(10)
        weights_in_forward_call = torch.rand(self.batch, self.dim, self.dim, self.dim).long()

        criterion = PixelWiseCrossEntropyLoss(weights)
        loss = criterion(self.predicted, self.target, weights_in_forward_call)
        print(loss)

    def test_TagsAngularLoss(self):
        criterion = TagsAngularLoss(classes=self.classes)
        pred = [self.predicted, self.predicted, self.predicted]
        target = [self.target, self.target, self.target]
        loss = criterion(pred, target)
        print("test_TagsAngularLoss", loss.item())

    def test_WeightedSmoothL1Loss(self):
        criterion = WeightedSmoothL1Loss(classes=self.classes)
        loss = criterion(self.predicted, self.target)
        loss.backward()

        assert (loss.item() > 0) and (loss.item() < 3)


test_crit = TestCriterion(batch=1, dim=64, classes=10)

test_crit.test_BCEDiceLoss()
test_crit.test_DiceLoss()
test_crit.test_generalized_diceLoss()

test_crit.test_WeightedCrossEntropyLoss()
test_crit.test_WeightedSmoothL1Loss()
test_crit.test_ContrastiveLoss()

test_crit.test_TagsAngularLoss()

# test_crit.test_PixelWiseCrossEntropyLoss()
