import numpy as np
import torch

from .apply_augmentations import Augment


class DictToNumpy(Augment):
    def __init__(self, modality_keys, apply_to_label=True):
        """
        Convert dict of np arrays to nparray
        Args:
            modality_keys ():
            apply_to_label ():
        """
        super(DictToNumpy, self).__init__(modality_keys, apply_to_label)

    def __call__(self, data):
        stack_list = []
        for key in self.modality_keys:
            if key is not 'label':
                stack_list.append(data[key])
        input_tensor = np.stack(stack_list)
        if self.apply_to_label:
            label_tensor = data['label']
            return input_tensor, label_tensor
        return input_tensor, None

class PrepareInput(Augment):
    def __init__(self, modality_keys, apply_to_label=True):
        """
        Convert dict of np arrays to nparray
        Args:
            modality_keys ():
            apply_to_label ():
        """
        super(PrepareInput, self).__init__(modality_keys, apply_to_label)

    def __call__(self, data):
        stack_list = []
        for key in self.modality_keys:
            if key is not 'label':
                stack_list.append(data[key])
        input_tensor = np.stack(stack_list)
        if self.apply_to_label:
            label_tensor = data['label']
            return torch.from_numpy(input_tensor), torch.from_numpy(label_tensor)
        return torch.from_numpy(input_tensor), None

class DictToTensor(Augment):
    def __init__(self, modality_keys, apply_to_label=True):
        """
        Convert dict of np arrays to nparray
        Args:
            modality_keys ():
            apply_to_label ():
        """
        super(DictToTensor, self).__init__(modality_keys, apply_to_label)
    """
    Converts the input image to a tensor
    """

    def __call__(self, data):
        """
        Apply the transform to `img` and make it contiguous.

        Args:
            img (numpy array):
        """
        for key in self.modality_keys:
            if torch.is_tensor(data[key]):
                data[key] = data[key].contiguous()
            else:
                data[key] = torch.as_tensor(np.ascontiguousarray(data[key]))
        return data

class DictToList(object):
    def __call__(self, data):
        return list(data.values())
