<div align="center">
<img src="figures/med-zoo-logo.png" width=662 height=250/>
</div>

# A 3D multi-modal medical image segmentation library in PyTorch
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/black0017/MedicalZooPytorch/blob/master/Quickstart_MedicalZoo.ipynb)
![](https://img.shields.io/github/license/black0017/MedicalZooPytorch)

We strongly believe in open and **reproducible deep learning research**. Our goal is to implement an open-source medical image segmentation library of state of the art 3D deep neural networks in PyTorch. We also implemented a bunch of data loaders of the most common medical image datasets. **The first stable release of our repository is almost ready.**  This project started as an MSc Thesis and is currently **under further development.** 

Although this work was initially focused on **3D multi-modal brain MRI segmentation** we are slowly adding more architectures and data-loaders. Stay tuned! **More** updates are coming...

## Quick Start
- If you want to quickly understand the fundamental concepts, we strongly advice to check our [blog post](https://theaisummer.com/medical-image-deep-learning/ "MedicalZooPytorch article"), which provides a high level overview of all the aspects of medical image segmentation and deep learning. 
- Alternatively, you can create a **virtual environment** and install the requirements. Check the installation folder for more instructions.
- You can also take a quick glance at the [manual](./manual/README.md).
- If you do not have a capable environment or device to run this projects then you could give Google Colab a try. It allows you to run the project using a GPU device, free of charge. You may try our Colab demo using this notebook:[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/black0017/MedicalZooPytorch/blob/master/Quickstart_MedicalZoo.ipynb)
## Implemented architectures

- #### [U-Net3D](https://arxiv.org/abs/1606.06650) Learning Dense Volumetric Segmentation from Sparse Annotation Learning Dense Volumetric Segmentation from Sparse Annotation

- #### [V-net](https://arxiv.org/abs/1606.04797) Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation

- #### [ResNet3D-VAE](https://arxiv.org/pdf/1810.11654.pdf) 3D MRI brain tumor segmentation using auto-encoder regularization

- #### [U-Net](https://arxiv.org/abs/1505.04597 "official paper") Convolutional Networks for Biomedical Image Segmentation

- #### [SkipDesneNet3D](https://arxiv.org/pdf/1709.03199.pdf) 3D Densely Convolutional Networks for Volumetric Segmentation

- #### [HyperDense-Net](https://arxiv.org/abs/1804.02967) A hyper-densely connected CNN for multi-modal image segmentation

- #### [multi-stream Densenet3D](https://arxiv.org/abs/1804.02967) A hyper-densely connected CNN for multi-modal image segmentation

- #### [DenseVoxelNet](https://arxiv.org/abs/1708.00573) Automatic 3D Cardiovascular MR Segmentation with Densely-Connected Volumetric ConvNets

- #### [MED3D](https://arxiv.org/pdf/1904.00625.pdf) Transfer learning for 3D medical image analysis

- #### [HighResNet3D](https://arxiv.org/pdf/1707.01992.pdf) On the Compactness, Efficiency, and Representation of 3D Convolutional Networks: Brain Parcellation as a Pretext Task

- #### [COVID-Net]( https://arxiv.org/pdf/2003.09871.pdf) A Tailored Deep Convolutional Neural Network Design for Detection of COVID-19 Cases from Chest Radiography Images

## Implemented medical imaging data-loaders

- #### [Iseg 2017](http://iseg2017.web.unc.edu/ "Official iseg-2017 dataset page")

- #### [Mrbrains 2018](https://mrbrains18.isi.uu.nl/ "Mrbrains 2018 official website")

- #### [MICCAI BraTs2018](https://www.med.upenn.edu/sbia/brats2018/data.html "Brain Tumor Segmentation Challenge 2018")

- #### [IXI brain development Dataset ](https://brain-development.org/ixi-dataset/  "IXI Dataset")

- #### [MICCAI Gleason 2019 Challenge](https://gleason2019.grand-challenge.org/ "MICCAI2019 Gleason challenge")

- #### [COVID-CT dataset](https://arxiv.org/pdf/2003.13865.pdf)

- #### [COVIDx dataset](https://github.com/IliasPap/COVIDNet/blob/master/README.md)
## Preliminary results

### Visual results on Iseg-2017

<img src="figures/comparison.png"/>

### Iseg and Mr-brains 
| Model | # Params (M) | [MACS](https://en.wikipedia.org/wiki/Multiply%E2%80%93accumulate_operation)(G) |   Iseg 2017 [DSC](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) (%) | Mr-brains 4 classes DSC (%) |
|:-----------------:|:------------:|:--------:|:-------------------:|:-------------------:|
|   Unet3D   |   17 M   |  0.9  |  **93.84**  | **88.61** |
|   Vnet   |   **45 M**   |  12  |   87.21 | 84.09 |
|   DenseNet3D   |   3 M   |   5.1   |  81.65 |79.85|
|   SkipDenseNet3D   |   1.5 M   |   **31**   |  - |-|
|  DenseVoxelNet  |   1.8 M   |   8    | - | - |
|  HyperDenseNet  |   10.4 M   |   5.8   | - | - |


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
-  The arguments that you can modify are extensively listed in the [manual](./manual/README.md).

## New released cool features (04/2020)

- On the fly 3D total volume visualization
- Tensorboard and PyTorch 1.4 support to track training progress
- Code cleanup and packages creation
- Offline sub-volume generation 
- Add Hyperdensenet, 3DResnet-VAE, DenseVoxelNet
- Fix mrbrains,Brats2018, IXI,MICCAI 2019 gleason challenge dataloaders
- Add confusion matrix support for understanding training dynamics
- Visualizations

## Top priorities
- [x] Unify/Generalize Train and Test functions
- [ ] Minimal test prediction example with pre-trained models
- [ ] Save produced 3d-total-segmentation as nifty files

## Support 
If you **really** like this repository and find it useful, please consider (★) **starring** it, so that it can reach a broader audience of like-minded people. It would be highly appreciated :) !

## Contributing to Medical ZOO
If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues. More info on the [contribute directory](./contribute/readme.md).

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


[contributors-shield]: https://img.shields.io/github/contributors/black0017/MedicalZooPytorch.svg?style=flat-square
[contributors-url]: https://github.com/black0017/MedicalZooPytorch/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/black0017/MedicalZooPytorch.svg?style=flat-square
[forks-url]: https://github.com/black0017/MedicalZooPytorch/network/members

[stars-shield]: https://img.shields.io/github/stars/black0017/MedicalZooPytorch.svg?style=flat-square
[stars-url]: https://github.com/black0017/MedicalZooPytorch/stargazers

[issues-shield]: https://img.shields.io/github/issues/black0017/MedicalZooPytorch.svg?style=flat-square
[issues-url]: https://github.com/black0017/MedicalZooPytorch/issues
