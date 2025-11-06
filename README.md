# EdgeTail
## Table of contents
- [1 Introduction](#1-introduction)
- [3 Supported models and datasets in different applications](#3-Supported-models-and-datasets-in-different-applications)
  * [3.1 Image classification](#31-Image-classification)
- [4 Supported Long-tail methods](#4-Supported-FL-methods)


## 1 Introduction


## 3 Supported models and datasets in different applications
### 3.1 Image classification
||Model Name|Data|Script|
|--|--|--|--|
|&nbsp; &nbsp; &nbsp; &nbsp;&#9745;&nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;[6 layer_CNN (NeurIPS'2020)](https://proceedings.neurips.cc/paper/2020/hash/258be18e31c8188555c2ff05b4d542c3-Abstract.html) &nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Cifar100](http://www.cs.toronto.edu/~kriz/cifar.html) &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp;&nbsp;<br>&nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp;&nbsp; &nbsp;[MiniImageNet](https://proceedings.neurips.cc/paper/2020/hash/258be18e31c8188555c2ff05b4d542c3-Abstract.html) &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;| &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;[Demo](scripts/models/sixcnn.sh) &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; |
|&nbsp; &nbsp; &nbsp; &nbsp;&#9745;&nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;[ResNet (CVPR'2016)](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) &nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;[MiniImageNet](https://image-net.org/download.php) &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp;&nbsp;<br>&nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp;&nbsp; &nbsp;[TinyImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip) &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;| &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;[Demo](scripts/models/ResNet.sh) &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;|
|&nbsp; &nbsp; &nbsp; &nbsp;&#9745;&nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;[MobileNetV2 (CVPR'2018)](https://openaccess.thecvf.com/content_cvpr_2018/html/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.html) &nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;[MiniImageNet](https://image-net.org/download.php) &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;|&nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;[Demo](scripts/models/MobileNet.sh) &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;|
|&nbsp; &nbsp; &nbsp; &nbsp;&#9745;&nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;[DenseNet(CVPR'2017)](https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf) &nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;[MiniImageNet](https://image-net.org/download.php) &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;|&nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;[Demo](scripts/models/DenseNet.sh)&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;|
|&nbsp; &nbsp; &nbsp; &nbsp;&#9745;&nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;[ViT(ICLR'2021)](https://iclr.cc/virtual/2021/oral/3458) &nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;[TinyImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip) &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;|&nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;[Demo](scripts/models/ViT.sh)&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;|
