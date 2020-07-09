import urllib.request
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

from lib.augment3D import *
from lib.visual3D_temp import show_mid_slice

url_1 = "https://nipy.org/nibabel/_downloads/c16214e490de2a223655d30f4ba78f15/someones_anatomy.nii.gz"
url_2 = "https://nipy.org/nibabel/_downloads/f76cc5a46e5368e2c779868abc49e497/someones_epi.nii.gz"

urllib.request.urlretrieve(url_1, './someones_epi.nii.gz')
urllib.request.urlretrieve(url_2, './someones_anatomy.nii.gz')

epi_img = nib.load('someones_epi.nii.gz')
anatomy_img = nib.load('someones_anatomy.nii.gz')
# convert to numpy
epi_img_numpy = epi_img.get_fdata()
anatomy_img_numpy = anatomy_img.get_fdata()

"""## 3D shapes and medical-header files"""
print(epi_img_numpy.shape)
print(anatomy_img_numpy.shape)
anat_header_file = anatomy_img.header
epi_img_numpy_header = epi_img.header
print(anat_header_file)
print("\n\n\n\n\n\n")
print(epi_img_numpy_header)



show_mid_slice(epi_img_numpy)
show_mid_slice(random_rotate3D(epi_img_numpy,-60,60))
show_mid_slice(random_rotate3D(epi_img_numpy,-60,60))


random_labels = np.zeros_like(epi_img_numpy)
ones = np.ones((10,10,10))
twos = np.ones((10,10,10))*2
random_labels[10:20,10:20,10:20] = ones
print("ok")
random_labels[22:32,22:32,22:32] = twos

show_mid_slice((epi_img_numpy))
show_mid_slice(random_crop_to_labels(epi_img_numpy, random_labels))
show_mid_slice(random_crop_to_labels(epi_img_numpy, random_labels))


show_mid_slice((epi_img_numpy))
show_mid_slice(random_flip(epi_img_numpy))
show_mid_slice(random_flip(epi_img_numpy))


show_mid_slice((epi_img_numpy))
show_mid_slice(random_shift(epi_img_numpy))
show_mid_slice(random_shift(epi_img_numpy))


show_mid_slice((epi_img_numpy))
show_mid_slice(random_zoom(epi_img_numpy))
show_mid_slice(random_zoom(epi_img_numpy))


show_mid_slice((epi_img_numpy))
show_mid_slice(elastic_transform_3d(epi_img_numpy))
show_mid_slice(elastic_transform_3d(epi_img_numpy))
show_mid_slice(elastic_transform_3d(epi_img_numpy))


print("RandomChoice test complete")
