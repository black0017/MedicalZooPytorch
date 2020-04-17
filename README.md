<div align="center">
<img src="figures/med-zoo-logo.png" width=662 height=250/>
</div>

# A 3D multi-modal medical image segmentation library in PyTorch

![](https://img.shields.io/github/license/black0017/MedicalZooPytorch)

Our goal is to implement an open-source medical image segmentation library of state of the art 3D deep neural networks in PyTorch along with data loaders of the most common medical MRI datasets. The first stable release of our repository is almost ready. We strongly believe in open and **reproducible deep learning research**. This project started as an MSc Thesis and is currently under further development.

## Beta release - work in progress!
Although this work was initially focused on **3D multi-modal brain MRI segmentation** we are slowly adding more architectures and data-loaders. Stay tuned! **More** updates are coming...

## Quick Start
- If you want to quickly understand the fundamental concepts we strongly advice to check our [blog post](https://theaisummer.com/medical-image-deep-learning/ "MedicalZooPytorch article"), which provides a high level overview of all the aspects of medical image segmentation and deep learning. 

- Alternatively, you can create a **virtual environment** and install the requirements. Check installation folder for more instructions.

## Implemented architectures

- #### [U-Net3D](https://arxiv.org/abs/1606.06650) Learning Dense Volumetric Segmentation from Sparse Annotation Learning Dense Volumetric Segmentation from Sparse Annotation

- #### [V-net](https://arxiv.org/abs/1606.04797) Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation

- #### [ResNet3D-VAE](https://arxiv.org/pdf/1810.11654.pdf) 3D MRI brain tumor segmentation using auto-encoder regularization

- #### [U-Net](https://arxiv.org/abs/1505.04597 "official paper") Convolutional Networks for Biomedical Image Segmentation

- #### [COVID-Net]( https://arxiv.org/pdf/2003.09871.pdf) A Tailored Deep Convolutional Neural Network Design for Detection of COVID-19 Cases from Chest Radiography Images

- #### [SkipDesneNet3D](https://arxiv.org/pdf/1709.03199.pdf) 3D Densely Convolutional Networks for Volumetric Segmentation

- #### [HyperDense-Net](https://arxiv.org/abs/1804.02967) A hyper-densely connected CNN for multi-modal image segmentation

- #### [multi-stream Densenet3D](https://arxiv.org/abs/1804.02967) A hyper-densely connected CNN for multi-modal image segmentation

- #### [DenseVoxelNet](https://arxiv.org/abs/1708.00573) Automatic 3D Cardiovascular MR Segmentation with Densely-Connected Volumetric ConvNets

- #### [MED3D](https://arxiv.org/pdf/1904.00625.pdf) TRANSFER LEARNING  FOR 3D MEDICAL IMAGE ANALYSIS

## Implemented data-loaders
- #### [Iseg 2017](http://iseg2017.web.unc.edu/ "Official iseg-2017 dataset page")

- #### [Mrbrains 2018](https://mrbrains18.isi.uu.nl/ "Mrbrains 2018 official website")

- #### [MICCAI Gleason 2019 Challenge](https://gleason2019.grand-challenge.org/ "MICCAI2019 Gleason challenge")

- #### [MICCAI BraTs2018](https://www.med.upenn.edu/sbia/brats2018/data.html "Brain Tumor Segmentation Challenge 2018")

- #### [ IXI brain development Dataset ](https://brain-development.org/ixi-dataset/  "IXI Dataset")

## Results

 To be updated **really really** soon......(this month)

## Usage

#### How to train your model 
- For Iseg-2017 :
```
python ./tests/train_iseg.py --args
```
- For MR brains 2018 (4 classes)
```
python ./tests/train_mrbrains_4_classes.py --args
```
- For MR brains 2018 (8 classes)
```
python ./tests/train_mrbrains_8_classes.py --args
```
- For MICCAI 2019 Gleason Challenge
```
python ./tests/test_miccai_2019.py --args
```
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

## New released cool features (04/2020)

- On the fly 3D total volume visualization
- Tensorboard and PyTorch 1.4 support to track training progress
- Code cleanup and packages creation
- Offline sub-volume generation 
- Add Hyperdensenet, 3DResnet-VAE, DenseVoxelNet
- Fix mrbrains,Brats2018, IXI,MICCAI 2019 gleason challenge dataloaders
- Add confusion matrix support for understanding training dynamics
- Write Tests for the project


## Top priorities
- [ ] Unify/Generalize Train and Test functions
- [ ] Test new architectures
- [ ] Minimal test prediction example with pre-trained models
- [ ] Save produced 3d-total-segmentation as nifty files

## Current team

#### [Ilias Papastatis](https://github.com/IliasPap "Git page" )

#### [Nikolas Adaloglou](https://www.linkedin.com/in/adaloglou17/ "LinkedIn page")


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
