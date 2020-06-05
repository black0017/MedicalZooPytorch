import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage.filters import gaussian_filter

"""
 Elastic deformation of images as described in
 Simard, Steinkraus and Platt, "Best Practices for
 Convolutional Neural Networks applied to Visual
 Document Analysis", in
 Proc. of the International Conference on Document Analysis and
 Recognition, 2003.

 Modified from:
 https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a
 https://github.com/fcalvet/image_tools/blob/master/image_augmentation.py#L62

 Modified to take 3D inputs
 Deforms both the image and corresponding label file
 Label volumes are interpolated via nearest neighbour 
 """


def elastic_transform_3d(img_numpy, labels=None, alpha=1, sigma=20, c_val=0.0, method="linear"):
    """
    :param img_numpy: 3D medical image modality
    :param labels: 3D medical image labels
    :param alpha: scaling factor of gaussian filter
    :param sigma: standard deviation of random gaussian filter
    :param c_val: fill value
    :param method: interpolation method. supported methods : ("linear", "nearest")
    :return: deformed image and/or label
    """
    assert img_numpy.ndim == 3, 'Wrong img shape, provide 3D img'
    if labels is not None:
        assert img_numpy.shape == labels.shape, "Shapes of img and label do not much!"
    shape = img_numpy.shape

    # Define 3D coordinate system
    coords = np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2])

    # Interpolated img
    im_intrps = RegularGridInterpolator(coords, img_numpy,
                                        method=method,
                                        bounds_error=False,
                                        fill_value=c_val)

    # Get random elastic deformations
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma,
                         mode="constant", cval=0.) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma,
                         mode="constant", cval=0.) * alpha
    dz = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma,
                         mode="constant", cval=0.) * alpha

    # Define sample points
    x, y, z = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    indices = np.reshape(x + dx, (-1, 1)), \
              np.reshape(y + dy, (-1, 1)), \
              np.reshape(z + dz, (-1, 1))

    # Interpolate 3D image image
    img_numpy = im_intrps(indices).reshape(shape)

    # Interpolate labels
    if labels is not None:
        lab_intrp = RegularGridInterpolator(coords, labels,
                                            method="nearest",
                                            bounds_error=False,
                                            fill_value=0)

        labels = lab_intrp(indices).reshape(shape).astype(labels.dtype)
        return img_numpy, labels

    return img_numpy


class ElasticTransform(object):
    def __init__(self, alpha=1, sigma=20, c_val=0.0, method="linear"):
        self.alpha = alpha
        self.sigma = sigma
        self.c_val = c_val
        self.method = method

    def __call__(self, img_numpy, label=None):
        return elastic_transform_3d(img_numpy, label, self.alpha, self.sigma, self.c_val, self.method)
