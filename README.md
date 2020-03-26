# Medical Zoo Pytorch
Our goal is to implementent an open-source medical image segmentation library of state of the art 3D deep neural networks in PyTorch along with data loaders of the most common medical MRI datasets. The first stable release of our repository is almost ready.

We strongly believe in open and reproducible deep learning research.
In order to reproduce our results, the code (alpha release) and materials of this work are available in this repository.

This project started as an MSc Thesis and is currently under further development. For 3D multi-modal brain MRI segmentation check the thesis branch of this repository.

## Beta release - work in progress
Although this work was initially focused on 3D multi-modal brain MRI segmentation we will slowly add more architectures and dataloaders.

## New released cool features (03/2020)

1. Batch size training support
2. On the fly volume vizualization
3. Tensorboard and PyTorch 1.4 support to track training progress
3. Code cleanup and package creation
4. Offline sub-volume generation 

## Top priorities

1. Fix Brats2018 and mrbrains dataloaders
2. Add hyper densenet and other architectures
3. Save produced 3d-segmenentation as medical image 


## Documentation
Check installation folder for instructions 


## Implemented dataloaders
ISEG 2017

MRBRAINS 2018

## Implemented architectures
Densenet3D (1-stream, 2-stream and 3-stream)

Unet3D

Vnet

## Support 
If you like this repository and find it useful, please consider (★) starring it, so that it can reach a broader audience.
