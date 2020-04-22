<div align="center">
<img src="../figures/med-zoo-logo.png" width=662 height=250/>
</div>

# Medical ZOO: Manual

![](https://img.shields.io/github/license/black0017/MedicalZooPytorch)

### Basics
- All models accept the input the channels(in_channels) and the segmentation classes(classes) as the first two parameters and produce un-normalized outputs
- All losses accept as input the pred in 5D shape of [batch,classes,dim_1,dim_2,dim_3] and 4D target shape of [batch, dim_1, dim_2, dim_3]. It is converted to one-hot inside the loss function for consistency reasons. Furthermore the normalization of the predictions is handled here. Dice-based losses return the scalar loss for backward(), and the prediction per channels in numpy to track training progress

  
### Arguments and explanation

- Arguments that you can modify
```
--batchSz, type=int, default=4, help='The batch size for training and validation'

--dim,  default=(64, 64, 64),  help='The sub-image or sub-volume that you want to crop for 2D specify as dim=(64, 64)' 

--nEpochs, type=int, default=250 ,  help='The training epochs'

--inChannels, type=int, choices=(1,2,3) , help='The desired modalities/channels that you want to use'

--inModalities, type=int, choices=(1,2,3), help='The modalities of the dataset'

--samples_train, type=int, default=10

--samples_val, type=int, default=10

--fold_id, default='1', type=str, help='Select subject for fold validation'

--lr, default=1e-3, type=float, help='learning rate (default: 1e-3)'

--cuda, default=True, help='whether you want to use cuda'

--model, type=str, default='UNET3D', choices=("RESNET3DVAE",'UNET3D',  'DENSENET1', 'DENSENET2', 'DENSENET3', 'HYPERDENSENET', "SKIPDENSENET3D",
                  "DENSEVOXELNET",'VNET','VNET2')

 --opt', type=str, default='sgd', choices=('sgd', 'adam', 'rmsprop')
```


### Models

| Model | # Params (M) | MACS(G) |
|:-----------------:|:------------:|:--------:|
|   Unet3D   |   17 M   |  0.9  |
|   Vnet   |   **45 M**   |  12  |
|   DenseNet3D   |   3 M   |   5.1   |
|   SkipDenseNet3D   |   1.5 M   |   **31**   |
|  DenseVoxelNet  |   1.8 M   |   8    |
|  HyperDenseNet  |   10.4 M   |   5.8   |


### Supported losses
- BCEDiceLoss
- GeneralizedDiceLoss
- DiceLoss
- WeightedSmoothL1Loss
- TagsAngularLoss
- ContrastiveLoss
- WeightedCrossEntropyLoss

### Medical Image utilities

- **rescale_data_volume(img_numpy, out_dim)** : Resize the 3d numpy array to the dim size,:param out_dim is the new 3d tuple
- **transform_coordinate_space(modality_1, modality_2)** : Accepts nifty objects. Transfers coordinate space from modality_2 to modality_1
- **normalize_intensity(img_tensor, normalization="mean")**: Accepts an image tensor and normalizes it.:param normalization: choices = "max", "mean" , type=str
- **random_rotate3D(img_numpy, min_angle, max_angle)**:  Returns a random rotated array in the same shape, :param img_numpy: 3D numpy array,:param min_angle: in degrees, param max_angle: in degrees
- **resample_to_output(img_nii, voxel_sizes=resample)** : reshamples voxel space


## License and citation
Advice the LICENSE.md file. For usage of third party libraries and repositories please advise the respective distributed terms. It would be nice to cite the original models and datasets. If you want, you can also cite this work as:

```
@MastersThesis{adaloglou2019MRIsegmentation,
author = {Adaloglou Nikolaos},
title={Deep learning in medical image analysis: a comparative analysis of
multi-modal brain-MRI segmentation with 3D deep neural networks},
school = {University of Patras},
note="\url{https://github.com/black0017/MedicalZooPytorch}",
year = {2019},
organization={Nemertes}}
```

## Support 
If you **really** like this repository and find it useful, please consider (★) **starring** it, so that it can reach a broader audience of like-minded people. It would be highly appreciated :) !
