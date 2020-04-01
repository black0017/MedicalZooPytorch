# Medical Zoo Pytorch
Our goal is to implementent an open-source medical image segmentation library of state of the art 3D deep neural networks in PyTorch along with data loaders of the most common medical MRI datasets. The first stable release of our repository is almost ready.

We strongly believe in open and **reproducible deep learning research**.
In order to reproduce our results, the code and materials of this work are available in this repository.

This project started as an MSc Thesis and is currently under further development. For 3D multi-modal brain MRI segmentation check the thesis branch of this repository.

## Beta release - work in progress!
Although this work was initially focused on **3D multi-modal brain MRI segmentation** we will slowly add more architectures and dataloaders.

## New released cool features (03/2020)

1. Batch size training support
2. On the fly 3D total volume vizualization
3. Tensorboard and PyTorch 1.4 support to track training progress
3. Code cleanup and package creation
4. Offline sub-volume generation 
5. Add hyperdensenet and 3dresnet-vae
6. Fix mrbrains dataloader
7. Add MICCAI 2019 gleason challenge
8. Add confusion matrix support for understaning training dynamics

## Top priorities
- [ ] Unify/Generalize Train and Test functions
- [x] Fix mrbrains dataloader(4 & 8 classes)
- [x] Fix Brats2018 dataloaders
- [x] Test new architectures
- [x] Minimal test pred example with pretrained models
- [x] Save produced 3d-total-segmenentation as nifti files  
- [ ] Test conf. matrix
- [ ] Test 3d-Vizualization memory occupation
- [ ] Tensorboard Logging
- [ ] Smart sampling


## Implemented dataloaders
[Iseg 2017](http://iseg2017.web.unc.edu/ "Official iseg-2017 dataset page")

[Mrbrains 2018](https://mrbrains18.isi.uu.nl/ "Mrbrains 2018 official website")

[Gleason 2019 Challenge](https://gleason2019.grand-challenge.org/ "MICCAI2019 Gleason challenge")

## Implemented architectures
[Densenet3D](https://arxiv.org/abs/1804.02967)

[U-Net3D](https://arxiv.org/abs/1606.06650)

[Vnet](https://arxiv.org/abs/1606.04797)

[Hyperdensenet](https://arxiv.org/abs/1804.02967)

[ResNet3D-VAE](https://arxiv.org/pdf/1810.11654.pdf)

[U-Net2D](https://arxiv.org/abs/1505.04597 "official paper")

## Results

#### To be updated **really really** soon......




## Documentation
Check installation folder for instructions. To be updated...

## Our current team
### [Ilias Papastatis](https://github.com/IliasPap "Git page" )

### [Nikolas Adaloglou](https://www.linkedin.com/in/adaloglou17/ "LinkedIn page")

## Support 
If you **really** like this repository and find it useful, please consider (★) **starring** it, so that it can reach a broader audience of like-minded people.
