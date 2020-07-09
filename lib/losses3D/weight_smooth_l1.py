import torch
from lib.losses3D.basic import expand_as_one_hot


# Code was adapted and mofified from https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/losses.py

class WeightedSmoothL1Loss(torch.nn.SmoothL1Loss):
    def __init__(self, threshold=0, initial_weight=0.1, apply_below_threshold=True,classes=4):
        super().__init__(reduction="none")
        self.threshold = threshold
        self.apply_below_threshold = apply_below_threshold
        self.weight = initial_weight
        self.classes = classes

    def forward(self, input, target):
        target = expand_as_one_hot(target,self.classes)
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"
        l1 = super().forward(input, target)

        if self.apply_below_threshold:
            mask = target < self.threshold
        else:
            mask = target >= self.threshold

        l1[mask] = l1[mask] * self.weight

        return l1.mean()
