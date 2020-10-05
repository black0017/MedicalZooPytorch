import nibabel as nib
import numpy as np
from nibabel.processing import resample_to_output
from scipy import ndimage


class MRIReader(object):
    def __call__(self, path):
        return NibToNumpy()(NibabelReader()(path))


class NibabelReader(object):
    def __call__(self, path):
        return nib.load(path)


class ToCanocical(object):
    def __call__(self, img_nii):
        return nib.as_closest_canonical(img_nii)


class Resample(object):
    def __init__(self, resample):
        self.resample = resample

    def __call__(self, img_nii):
        if self.resample is not None:
            return resample_to_output(img_nii, voxel_sizes=self.resample)
        return img_nii


class NibToNumpy(object):
    def __call__(self, img_nii):
        return np.squeeze(img_nii.get_fdata(dtype=np.float32))


class Rescale(object):
    def __init__(self, out_dim):
        self.out_dim = out_dim

    def __call__(self, img_numpy):
        """Resize the 3d numpy array to the dim size

        Args:
            out_dim is the new 3d tuple
        """
        depth, height, width = img_numpy.shape
        scale = [self.out_dim[0] * 1.0 / depth, self.out_dim[1] * 1.0 / height, self.out_dim[2] * 1.0 / width]
        return ndimage.interpolation.zoom(img_numpy, scale, order=0)
