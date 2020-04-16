<div align="center">
<img src="../figures/med-zoo-logo.png" width=662 height=250/>
</div>

# Medical ZOO: Manual

![](https://img.shields.io/github/license/black0017/MedicalZooPytorch)

### Basics
- All models accept the input the channels(in_channels) and the segmentation classes(classes) as the first two parameters and produce un-normalized outputs
- All losses accept as input the pred in 5D shape of [batch,classes,dim_1,dim_2,dim_3] and 4D target shape of [batch, dim_1, dim_2, dim_3]. It is converted to one-hot inside the loss function for consistency reasons. Furthermore the normalization of the predictions is handled here. Dice-based losses return the scalar loss for backward(), and the prediction per channels in numpy to track training progress

  
### Arguments and explanation


### Models


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
