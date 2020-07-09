<div align="center">
<img src="../figures/med-zoo-logo.png" width=662 height=250/>
</div>

# Medical ZOO: Manual

![](https://img.shields.io/github/license/black0017/MedicalZooPytorch)

### Basics
- All models accept two parameters: a) the input the channels (in_channels), and b) the segmentation classes (classes) and produce **un-normalized** outputs
- All losses accept as input the prediction in 5D shape of [batch,classes,dim_1,dim_2,dim_3] and the target in 4D target shape of [batch, dim_1, dim_2, dim_3]. It is converted to one-hot inside the loss function for consistency reasons. Furthermore the normalization of the predictions is handled here. Dice-based losses return the scalar loss for backward(), **and** the prediction per channels in numpy to track training progress

  
### Arguments and explanation

- Arguments that you can modify can be found below:
```
--batchSz, type=int, default=4, help='The batch size for training and validation'
--dataset_name, type=str, default="iseg2017", choices=('iseg2017','brats2018','brats2019','iseg2019','mrbrains4','mrbrains9','miccai2019')

--dim,  default=(64, 64, 64),  help='The sub-image or sub-volume that you want to crop for 2D specify as dim=(64, 64)' 

--nEpochs, type=int, default=250 ,  help='The training epochs'

--inChannels, type=int, choices=(1,2,3) , help='The desired modalities/channels that you want to use'

--inModalities, type=int, choices=(1,2,3), help='The modalities of the dataset'

--samples_train, type=int, default=10

--samples_val, type=int, default=10

'--split', default=0.9, type=float, help='Select percentage of training data(default: 0.9)')

--lr, default=1e-3, type=float, help='learning rate (default: 1e-3)'

--cuda, default=True, help='whether you want to use cuda'

--model, type=str, default='UNET3D', choices=("RESNET3DVAE",'UNET3D',  'DENSENET1', 'DENSENET2', 'DENSENET3', 'HYPERDENSENET', "SKIPDENSENET3D",
                  "DENSEVOXELNET",'VNET','VNET2')
                  
--log_dir', type=str,     default='../runs/'

--opt', type=str, default='sgd', choices=('sgd', 'adam', 'rmsprop')
```


### Models (more to be added...)

| Model | # Params (M) | MACS (G) |
|:-----------------:|:------------:|:--------:|
|   Unet3D   |   17 M   |  0.9  |
|   Vnet   |   **45 M**   |  12  |
|   DenseNet3D   |   3 M   |   5.1   |
|   SkipDenseNet3D   |   1.5 M   |   **31**   |
|  DenseVoxelNet  |   1.8 M   |   8    |
|  HyperDenseNet  |   10.4 M   |   5.8   |


### Supported 3D losses
- BCE Dice Loss
- Generalized Dice Loss
- Dice Loss
- Weighted Smooth L1 Loss
- Tags Angular Loss
- Contrastive Loss
- Weighted Cross Entropy Loss

### Medical Image preprocessing utilities

- **rescale_data_volume(img_numpy, out_dim)** : Resize the 3d numpy array to the dim size,:param out_dim is the new 3d tuple
- **transform_coordinate_space(modality_1, modality_2)** : Accepts nifty objects. Transfers coordinate space from modality_2 to modality_1
- **normalize_intensity(img_tensor, normalization="mean")**: Accepts an image tensor and normalizes it.:param normalization: choices = "max", "mean" , type=str
- **resample_to_output(img_nii, voxel_sizes)** : re-samples voxel space



## Datasets

In the **next version** we will provide datasets that will be directly downloaded from google drive for the medical decathlon challenge. Stay tuned.

### [2018 MICCAI Medical Segmentation Decathlon](http://medicaldecathlon.com/)

Recent official results can be found [here](https://decathlon-10.grand-challenge.org/evaluation/results/).

|Task|Data Info/ Modalities| Train/Test | Volume size | Classes | Dataset size (GB)|
|---|---|---|---|---|---|
|1. Brats|Multi-modal MRI data (FLAIR, T1w, T1gd,T2w)| **484** / **266** |-|-| - |
|2. Heart|Mono-modal MRI |20 / 10 |-|-|-|
|3. Hippocampus head and body|Mono-modal MRI | **263** / **131** |-|-|-|
|4. Liver & Tumor|Portal venous phase CT | 131 / 70 |-|-|-|
|5. Lung|CT |64 / 32|-|-|-|
|6. Pancreas & Tumor|Portal venous phase CT |**282** / **139** |-|-|-|
|7. Prostate central gland and peripheral|Multi-modal MRI (T2, ADC) |32 / 16| -           |-|-|
|8. Hepatic vessel & Tumor| CT |**303** / **140**|-|-|-|
|9. Spleen|CT |41 / 20|-|-|-|
|10. Colon|CT |41 / 20|-|-|-|

## Multi modal brain MRI datasets

|Task|Data Info/ Modalities| Train/Test | Volume size | Classes | Dataset size (GB)|
|---|---|---|---|---|---|
| Iseg 2017| T1, T2 | 10 / 10    |-|4| - |
| Iseg 2019| T1, T2 | 10 / 30    |-|4| - |
| BraTS 2018 |FLAIR, T1w, T1gd,T2w |20 / - |-|9 or 4|-|
| BraTS 2019 |FLAIR, T1w, T1gd,T2w |20 / - |-|9 or 4|-|
| MrBrains |FLAIR, T1w, T1gd,T2w |20 / - |-|9 or 4|-|
|IXI| T1,T2 **no labels** |  |-|-|-|

## 2D Medical imaging Datasets

|Task|Data Info/ Modalities| Train/Test | Volume size | Classes | Dataset size (GB)|
|---|---|---|---|---|---|
| |  | -  |-|-| - |
| - |- |- |-|-|-|
|-| - |  |-|-|-|

## Supported 3D augmentations

- Random rotate
- 3D Elastic deformation
- Random shift/translate
- Random scaling-zoom in/out
- Random crop
- Random axis flip

## License and citation
Advice the LICENSE.md file. For usage of third party libraries and repositories please advise the respective distributed terms. If you want, you can also cite this work as:

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