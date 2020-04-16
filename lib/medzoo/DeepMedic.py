"""
https://arxiv.org/pdf/1603.05959v3.pdf

Multi-scale 3D CNN with two convolutional pathways.
The kernels of the two pathways are here of size 53
(for illustration only to reduce the number  of  layers  in  the  figure).
The  neurons  of  the  last  layers  of  the  two pathways thus have receptive
fields of size 173 voxels.  The inputs of the two pathways are centered at the
same image location, but the second segmentis  extracted  from  a  down-sampled
version  of  the  image  by  a  factor  of  3.The second pathway processes
context in an actual area of size 513 voxels.DeepMedic,  our  proposed  11-layers
architecture,  results  by  replacing  each layer of the depicted pathways with
two that use 33kernels (see Sec. 2.3).Number of FMs and their size depicted as (Number×Size).

In order to incorporate both local and larger contextual information into our  3D  CNN,
we  add  a  second  pathway  that  operates  on  down-sampled images.  Thus, our dual
pathway 3D CNN simultaneously processes the input image at multiple scales (Fig. 5).
Higher level features such as the location within the brain are learned in the second pathway,
while the detailed local appearance of structures is captured in the first.  As the two pathways
are decoupled  in  this  architecture,  arbitrarily  large  context  can  be
processed by  the  second  pathway  by  simply  adjusting  the  down-sampling
factorFD.The  size  of  the  pathways  can  be  independently  adjusted  according
to  the computational capacity and the task at hand, which may require relatively more or
less filters focused on the down-sampled context.
"""
# TODO