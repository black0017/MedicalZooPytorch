import numpy as np
import matplotlib.pyplot as plt
real = np.load('predicted_maps/unet_3d_predMaps.npy')
real1 = np.load('predicted_maps/unet_3d_GT.npy')
print(real1.shape)
import nibabel as nib

gt = nib.load('./data/iseg/train/subject-1-label.img').get_fdata()
data = np.asarray(gt).astype(float)[77,:,:,0]


print("sss ",data.shape)
def show_slices(slices):
  """ Function to display row of image slices """
  fig, axes = plt.subplots(1, len(slices))
  for i, slice in enumerate(slices):
     axes[i].imshow(slice, cmap="gray", origin="lower")

def show_full_slice(slices):
  """ Function to display row of image slices """
  fig, axes = plt.subplots(1, len(slices))
  for i, slice in enumerate(slices):
     axes[i].imshow(slice, cmap="gray", origin="lower")

#print(data)
print(np.where(data>100,200,data))
data = np.where((data==10.0),60,data)
print(np.max(real),real.dtype)
slices = []
slices.append(data)
for i in range(4):
    slices.append(real[i,0,32,:,:])
    print(slices[i].shape)
#slices.append()
#show_slices([real1[0,32,:,:],real[3,0,32,:,:]])

#plt.show()
show_slices(slices)

plt.show()
slices = []
for i in range(4):
    slices.append(real[i,0,32,:,:])
    print(slices[i].shape)
final_img = np.zeros((256,192))
print(final_img.shape,final_img[0:128,0:64].shape,final_img[127:255,63:191].shape)




final_img[63:191,63:191] = slices[3][:,:]
final_img[0:128,63:191] = slices[2][:,:]
final_img[0:128,0:64] = slices[0][:,63:127]
final_img[63:191,0:64] = slices[1][:,63:127]
#data =  np.transpose(data,(1,0))
#final_img[0:128,0:128] = slices[0]
#final_img[127:255,127:191] = slices[0][:,63:127]



show_slices([data,final_img])
plt.show()


'''
slice_0 = slice_0.reshape(64,-1)
slice_1 = slice_1.reshape(64,-1)
slices = np.concatenate((slice_1,slice_0),axis=1)
print(slices.shape)
slices = slices.reshape(64,256,192)
print(slices.shape)
#slice_0x,slice_0y = slice_0[0]
#c = np.vstack((slice_0,slice_1))
#print(c.shape)
'''
b = np.concatenate((slice_1,slice_0),axis=1)
#print(b.shape)
show_slices([slice_1,final_img])
plt.show()
