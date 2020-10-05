import numpy as np
import scipy.ndimage as ndimage
from .apply_augmentations import RandomAugment
def random_zoom(img_numpy, sample,min_percentage=0.8, max_percentage=1.1):
    """[summary]

    Args:
        img_numpy : [description]
        min_percentage : [description]. Defaults to 0.8.
        max_percentage : [description]. Defaults to 1.1.

    Returns:
        zoom in/out aigmented img
    """    
   
    z = sample * (max_percentage - min_percentage) + min_percentage
    zoom_matrix = np.array([[z, 0, 0, 0],
                            [0, z, 0, 0],
                            [0, 0, z, 0],
                            [0, 0, 0, 1]])
    return ndimage.interpolation.affine_transform(img_numpy, zoom_matrix)


class RandomZoom(RandomAugment):
    """

    """
    def __init__(self, modality_keys,apply_to_label,min_percentage=0.8, max_percentage=1.1):
        super(RandomZoom, self).__init__(modality_keys,apply_to_label)
        self.min_percentage = min_percentage
        self.max_percentage = max_percentage
        self.random = np.random.sample()

    def get_params(self):
        return np.random.sample()
    def __call__(self, data):

        randomness = self.get_params()
        for key in self.modality_keys:
            if key!='label':
                data[key] = random_zoom( data[key],randomness, self.min_percentage, self.max_percentage)
        if self.apply_to_label:
            data['label'] = random_zoom(data['label'], randomness, self.min_percentage, self.max_percentage)

        return data
