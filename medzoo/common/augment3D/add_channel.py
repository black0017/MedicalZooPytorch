import numpy as np

from .apply_augmentations import Augment

class AddChannelDim(Augment):
    def __init__(self, modality_keys, apply_to_label = False):
        """
        Adds channel as first dimension (expand)
        Args:
            modality_keys ():
            apply_to_label ():
        """
        super(AddChannelDim, self).__init__(modality_keys, apply_to_label)

    def __call__(self,data):
        for key in self.modality_keys:

            if key != 'label':
                data[key] = np.expand_dims(data[key],axis=0)
            if self.apply_to_label:
                data['label'] = np.expand_dims(data['label'], axis=0)
        return data