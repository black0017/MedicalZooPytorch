import urllib.request
from lib.medloaders.medical_image_process import *
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
"""# Lets see some slices"""

show_mid_slice(epi_img_numpy)
show_mid_slice(anatomy_img_numpy)

print(epi_img.shape)
print(anatomy_img.shape)
transformed = transform_coordinate_space(epi_img, anatomy_img)
print("Transormed shape", transformed.shape)

"""# Spliting and unifing transformations"""

transform_A = nib.affines.from_matvec(np.diag([2, 3, 4]), [9, 10, 11])
translation_A, rotation_A = nib.affines.to_matvec(transform_A)
print(translation_A, rotation_A)

voxels_out = nib.affines.voxel_sizes(transform_A)
print(voxels_out)

"""# Medical Image interpolation"""

show_mid_slice(epi_img_numpy)
print(epi_img_numpy.shape)



print(epi_img_numpy.ndim)
result = rescale_data_volume(epi_img_numpy, (32, 32, 32))
print(result.shape)
show_mid_slice(result)


# clip_range ?????
# outliers
epi_img_numpy[10,10,10] = 300
epi_img_numpy[10,10,11] = -300


anatomy_img_numpy[10,10,10] = 300
anatomy_img_numpy[10,10,11] = -300

show_mid_slice((epi_img_numpy))
show_mid_slice(clip_range(epi_img_numpy))

show_mid_slice((anatomy_img_numpy))
show_mid_slice(clip_range(anatomy_img_numpy))



