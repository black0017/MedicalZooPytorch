import torch.nn as nn
import torch


class DiceLoss(nn.Module):
    """
    Computes multi channel Dice Loss
    The output from the network during training is assumed to be un-normalized probabilities
    """

    def __init__(self, all_classes=4, desired_classes=0, epsilon=1e-5, sigmoid_normalization=True, ):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
        self.ignore = False
        self.all_classes = all_classes
        self.desired_classes = desired_classes
        self.flag = (desired_classes != 0)
        print(self.flag)

        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)

    def expand_as_one_hot(self, input):
        """
        Converts NxDxHxW label image to NxCxDxHxW, where each label gets converted to its corresponding one-hot vector
        :param input: 4D input image (NxDxHxW)
        :param N: batch size
        :param C: number of channels/labels/classes
        :return: 5D output image (NxCxDxHxW)
        """
        assert input.dim() == 4
        desired_class_dimension = 1
        desired_class_value = 1

        inp_shape = input.size()  # 4D input image (NxDxHxW)
        src = input.unsqueeze(desired_class_dimension)

        # Creates desired shape 5D input image (NxCxDxHxW) in a list
        out_shape_list = list(inp_shape)
        out_shape_list.insert(desired_class_dimension, self.all_classes)
        desired_shape = tuple(out_shape_list)
        assert src.dim() == len(desired_shape)
        target = torch.zeros(desired_shape).to(input.device).scatter_(desired_class_dimension, src,
                                                                      desired_class_value)
        if self.flag:
            target = target.clone()[:, 0:self.desired_classes, ...]
        return target

    def compute_per_channel_dice(self, prediction, target):
        epsilon = 1e-5
        batch, classes, d1, d2, d3 = prediction.shape
        target = self.expand_as_one_hot(target.long())

        assert prediction.size() == target.size(), " 'prediction' and 'target' must have the same shape"

        prediction = prediction.view(batch, classes, -1)
        target = target.view(batch, classes, -1).float()

        # Compute per channel Dice Coefficient
        intersect = (prediction * target).sum(2)
        denominator = (prediction + target).sum(2)
        results = 2. * intersect / denominator.clamp(min=epsilon)
        return results

    def forward(self, prediction, target):
        prediction = self.normalization(prediction)
        per_channel_dice = self.compute_per_channel_dice(prediction, target)
        total_loss = torch.mean(1. - per_channel_dice)

        # TODO test
        with torch.no_grad():
            temp = per_channel_dice.cpu().detach().clone()
            scores = torch.mean(temp, dim=0).numpy()  # mean batch score statistics
        return total_loss, scores
