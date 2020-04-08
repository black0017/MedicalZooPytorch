# Medical Zoo Pytorch
Our goal is to implementent an open-source medical image segmentation library of state of the art 3D deep neural networks in PyTorch along with data loaders of the most common medical MRI datasets. The first stable release of our repository is almost ready. We strongly believe in open and **reproducible deep learning research**. This project started as an MSc Thesis and is currently under further development.

## Beta release - work in progress!
Although this work was initially focused on **3D multi-modal brain MRI segmentation** we are slowly adding more architectures and dataloaders. Stay tuned!

## Quick Start
- If you want to quickly understand the foundamental concepts we strongly advice to check our [blog post](https://theaisummer.com/medical-image-deep-learning/ "MedicalZooPytorch article"), which provides a high level overview of all the aspects of medical image segmentation and deep learning. 

- Alternatively, you can create a virtual enviroment and install the requirements. Check installation folder for more instructions.

## Implemented architectures

#### [U-Net3D](https://arxiv.org/abs/1606.06650) Learning Dense Volumetric Segmentation from Sparse Annotation Learning Dense Volumetric Segmentation from Sparse Annotation

#### [V-net](https://arxiv.org/abs/1606.04797) Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation

#### [ResNet3D-VAE](https://arxiv.org/pdf/1810.11654.pdf) 3D MRI brain tumor segmentation usingautoencoder regularization

#### [U-Net](https://arxiv.org/abs/1505.04597 "official paper") Convolutional Networks for Biomedical Image Segmentation

#### [COVID-Net]( https://arxiv.org/pdf/2003.09871.pdf) A Tailored Deep Convolutional Neural Network Design forDetection of COVID-19 Cases from Chest Radiography Images

#### [SkipDesneNet3D](https://arxiv.org/pdf/1709.03199.pdf) 3D Densely Convolutional Networks for VolumetricSegmentation

#### [HyperDense-Net](https://arxiv.org/abs/1804.02967) A hyper-densely connected CNN for multi-modal image segmentation

#### [multi-stream Densenet3D](https://arxiv.org/abs/1804.02967) A hyper-densely connected CNN for multi-modal image segmentation

#### [DenseVoxelNet](https://arxiv.org/abs/1708.00573) Automatic 3D Cardiovascular MR Segmentation with Densely-Connected Volumetric ConvNets

## Implemented dataloaders
[Iseg 2017](http://iseg2017.web.unc.edu/ "Official iseg-2017 dataset page")

[Mrbrains 2018](https://mrbrains18.isi.uu.nl/ "Mrbrains 2018 official website")

[Gleason 2019 Challenge](https://gleason2019.grand-challenge.org/ "MICCAI2019 Gleason challenge")

## Results

 To be updated **really really** soon......

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
- Arguments that ypu can modify
```
--batchSz, default=4

--dim,  default=(64, 64, 64)

--nEpochs, default=250

--inChannels, values=1,2,3

--inModalities, values=1,2,3

--fold_id, default='1', type=str, help='Select subject for fold validation')

--lr, default=1e-3, type=float, help='learning rate (default: 1e-3)')

--cuda, default=True)

--model, type=str, default='UNET3D', choices=('VNET', 'VNET2', 'UNET3D', 'DENSENET1', 'DENSENET2', 'DENSENET3','HYPERDENSENET'))

 --opt', type=str, default='sgd', choices=('sgd', 'adam', 'rmsprop'))
```

## New released cool features (03/2020)

1. Batch size training support
2. On the fly 3D total volume vizualization
3. Tensorboard and PyTorch 1.4 support to track training progress
3. Code cleanup and package creation
4. Offline sub-volume generation 
5. Add Hyperdensenet, 3DResnet-VAE, DenseVoxelNet
6. Fix mrbrains dataloader
7. Add MICCAI 2019 gleason challenge
8. Add confusion matrix support for understaning training dynamics


## Top priorities
- [ ] Unify/Generalize Train and Test functions
- [x] Fix mrbrains dataloader(4 & 8 classes)
- [ ] Fix Brats2018 dataloaders
- [x] Test new architectures
- [x] Minimal test pred example with pretrained models
- [x] Save produced 3d-total-segmenentation as nifti files
- [ ] Test conf. matrix

## Current team

#### [Ilias Papastatis](https://github.com/IliasPap "Git page" )

#### [Nikolas Adaloglou](https://www.linkedin.com/in/adaloglou17/ "LinkedIn page")


## License
Advice the LICENSE.md file. For usage of third party libraries and repositories please advise the respective distributed terms.

## Support 
If you **really** like this repository and find it useful, please consider (★) **starring** it, so that it can reach a broader audience of like-minded people. It would be highly appreciated :)
