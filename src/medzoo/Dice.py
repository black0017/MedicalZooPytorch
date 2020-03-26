import torch.nn as nn
import torch
import torch.optim as optim


class DiceLoss(nn.Module):
    """Computes Dice Loss, which just 1 - DiceCoefficient described above.
    Additionally allows per-class weights to be provided.
            # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify sigmoid_normalization=False.
    """

    def __init__(self, epsilon=1e-5, sigmoid_normalization=True,
                 skip_last_target=False, idx_to_ignore_after=None):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
        if idx_to_ignore_after is not None:
            self.idx_to_ignore_after = idx_to_ignore_after
        else:
            self.idx_to_ignore_after = None

        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)
        # if True skip the last channel in the target
        self.skip_last_target = skip_last_target

    def flatten(self, tensor):
        """Flattens a given tensor such that the channel axis is first.
        The shapes are transformed as follows:
           (N, C, D, H, W) -> (C, N * D * H * W)
        """
        C = tensor.size(1)
        # new axis order
        axis_order = (1, 0) + tuple(range(2, tensor.dim()))
        # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
        transposed = tensor.permute(axis_order)
        # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
        return transposed.view(C, -1)

    def expand_as_one_hot(self, input, C, ignore_index=None):
        """
        Converts NxDxHxW label image to NxCxDxHxW, where each label gets converted to its corresponding one-hot vector
        :param input: 4D input image (NxDxHxW)
        :param N: batch size
        :param C: number of channels/labels/classes
        :param ignore_index: ignore index to be kept during the expansion
        :return: 5D output image (NxCxDxHxW)
        """
        assert input.dim() == 4
        desired_class_dimension = 1
        desired_class_value = 1

        inp_shape = input.size()  # 4D input image (NxDxHxW)
        src = input.unsqueeze(desired_class_dimension)

        # creates desired shape 5D input image (NxCxDxHxW) in a list
        out_shape_list = list(inp_shape)
        out_shape_list.insert(desired_class_dimension, C)
        desired_shape = tuple(out_shape_list)

        assert src.dim() == len(desired_shape)

        if ignore_index is not None:
            # create ignore_index mask for the result
            expanded_src = src.expand(desired_shape)
            mask = expanded_src == ignore_index
            # clone the src tensor and zero out ignore_index in the input
            src = src.clone()
            src[src == ignore_index] = 0
            # scatter to get the one-hot tensor
            result = torch.zeros(desired_shape).to(input.device).scatter_(1, src, 1)
            # bring back the ignore_index in the result
            result[mask] = ignore_index
            return result
        else:
            return torch.zeros(desired_shape).to(input.device).scatter_(desired_class_dimension, src,
                                                                        desired_class_value)

    def compute_per_channel_dice(self, input, target, classes=4):
        epsilon = 1e-5
        batch, classes, d1, d2, d3 = input.shape
        # assumes that input is a normalized probability

        # input and target shapes must match
        target = self.expand_as_one_hot(target.long(), classes)
        if self.idx_to_ignore_after is not None:
            target = target.clone()[:, :self.idx_to_ignore_after, ...]
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        # input = self.flatten(input)
        # target = self.flatten(target)
        input = input.view(batch, classes, -1)
        target = target.view(batch, classes, -1).float()

        # Compute per channel Dice Coefficient
        intersect = (input * target).sum(2)
        denominator = (input + target).sum(2)
        results = 2. * intersect / denominator.clamp(min=epsilon)
        return results

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)
        per_channel_dice = self.compute_per_channel_dice(input, target)
        temp = per_channel_dice.clone().cpu()
        DSC = torch.mean(temp.detach(), dim=0).numpy()

        # Average the Dice score across all channels/classes
        per_channel_loss = 1. - per_channel_dice

        # total_loss = (per_channel_loss*torch.Tensor([0.1 , 0.3 , 0.3 , 0.3]))
        total_loss = torch.mean(per_channel_loss)
        return total_loss, DSC



