# Medical Zoo Pytorch Alpha release (Master Thesis project)
We strongly believe in open and reproducible deep learning research. In order to reproduce our results, the code (alpha release) and materials of this thesis are available in this repository. The goal is to implementent an open-source medical segmentation library of state of the art 3D deep neural networks in PyTorch along with data loaders of the most common medical MRI datasets.

For an updated version check the master branch.

## Alpha release - work in progress
![Alt text](./figs/intro.png?raw=true "title")

This MSc Thesis is focused on multi-modal brain segmentation.   For our experiments, we used two common benchmark datasets(ISEG 2017 & MRBRAINS2018) from MICCAI MRI image challenges. Brain MR segmentation challenges aim to evaluate state-of-the-art methods for the segmentation of brain by providing a 3D MRI dataset with ground truth tumor segmentation labels annotated by physicians. We perform a comparative analysis of the modern 3D architectures through extensive evaluations. The implemented networks were based on the specifications of the original papers. Finally, we discuss the reported results and provide future directions.

## Early Thesis results

## Comparison with Ground truth data
![Alt text](./comparison.png?raw=true "Dice coeff.")


### Train-Val curves
![Alt text](./figs/unet_3d_dice_coeff.jpg?raw=true "Dice coeff.")


![Alt text](./figs/unet_3d_loss.jpg?raw=true "Dice loss")

### Out volume features visualization
![Alt text](./figs/a1.png?raw=true "Slice viz")

![Alt text](./figs/a2.png?raw=true "Slice viz")

![Alt text](./figs/a3.png?raw=true "Slice viz")

## Virtual env
This command should get you working. Better Documentation in the master branch.
```
pip install torch numpy nibabel
```

## Where to brain-MRI segmentation datasets

### iSeg-2017
```
http://iseg2017.web.unc.edu/
```
### MR-BRAINS 2018 DATASET
```
https://mrbrains18.isi.uu.nl/
```


## Top References
```
[1] Çiçek, Ö., Abdulkadir, A., Lienkamp, S. S., Brox, T., & Ronneberger, O. (2016, October). 3D U-Net: learning dense volumetric segmentation from sparse annotation. In International conference on medical image computing and computer-assisted intervention (pp. 424-432). Springer, Cham.

[2] Dolz, J., Gopinath, K., Yuan, J., Lombaert, H., Desrosiers, C., & Ayed, I. B. (2018). HyperDense-Net: A hyper-densely connected CNN for multi-modal image segmentation. IEEE transactions on medical imaging, 38(5), 1116-1126.

[3] Milletari, F., Navab, N., & Ahmadi, S. A. (2016, October). V-net: Fully convolutional neural networks for volumetric medical image segmentation. In 2016 Fourth International Conference on 3D Vision (3DV) (pp. 565-571). IEEE.

[4] Wang, L., Nie, D., Li, G., Puybareau, É., Dolz, J., Zhang, Q., ... & Thung, K. H. (2019). Benchmark on automatic 6-month-old infant brain segmentation algorithms: the iSeg-2017 challenge. IEEE transactions on medical imaging.

```
