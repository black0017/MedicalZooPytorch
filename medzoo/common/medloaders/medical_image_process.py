import nibabel as nib
from PIL import Image
from nibabel.processing import resample_to_output

from medzoo.common.medloaders.preprocessing import *

"""
Functionalities for loading and preprocessing 2D/3D images
"""


def load_medical_image(path, type=None, resample=None,
                       viz3d=False, to_canonical=False, rescale=None, normalization='full_volume_mean',
                       clip_intenisty=[1, 99], crop_size=(0, 0, 0), crop=(0, 0, 0)):
    """

    Args:
        path:
        type:
        resample:
        viz3d:
        to_canonical:
        rescale:
        normalization:
        clip_intenisty:
        crop_size:
        crop:

    Returns:

    """
    # TODO check with lias
    if viz3d:
        return torch.from_numpy(np.squeeze(nib.load(path).get_fdata(dtype=np.float32)))

    img_nii = nifty_preprocessing(path, to_canonical, resample)
    img_np = img_nii.get_fdata(dtype=np.float32)

    # 1. Intensity outlier clipping
    if clip_intenisty and type != "label":
        img_np = percentile_clip(img_np)

    # 2. Rescale to specified output shape
    if rescale is not None:
        # TODO pytorch interpolation function
        rescale_data_volume(img_np, rescale)

    img_tensor = torch.from_numpy(img_np)

    # 3. intensity normalization
    if type != "label":
        img_tensor = normalize_intensity(img_tensor, normalization=normalization)

    # 4. cropping (for patch size-based training)
    img_tensor = crop_img(img_tensor, crop_size, crop)

    return img_tensor


def nifty_preprocessing(path, to_canonical, resample):
    img_nii = nib.load(path)
    if to_canonical:
        img_nii = nib.as_closest_canonical(img_nii)
    if resample is not None:
        img_nii = resample_to_output(img_nii, voxel_sizes=resample)
    return img_nii


def load_affine_matrix(path):
    """
    Reads an path to nifti file and returns the affine matrix as numpy array 4x4
    """
    img = nib.load(path)
    return img.affine


def load_2d_image(img_path, resize_dim=0, type='RGB'):
    """
    Functionality for 2D image loading.
    Uses pillow under the hood.

    Args:
        img_path: 2D image path
        resize_dim (int): dimension to resize the img
        type: as in Pillow i.e. 'RGB'

    Returns: numpy array of the resized img

    """
    image = Image.open(img_path)
    if type == 'RGB':
        image = image.convert(type)
    else:
        raise NotImplementedError

    if resize_dim != 0:
        image = image.resize(resize_dim)
    return np.array(image)
