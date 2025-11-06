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
|&nbsp; &nbsp; &nbsp; &nbsp;&#9745;&nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;[ResNet (CVPR'2016)](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) &nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;[Cifar100](http://www.cs.toronto.edu/~kriz/cifar.html) &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp;&nbsp;<br>&nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp;&nbsp; &nbsp;[TinyImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip) &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;| &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;[Demo](scripts/models/ResNet.sh) &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;|
|&nbsp; &nbsp; &nbsp; &nbsp;&#9745;&nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;[ViT(ICLR'2021)](https://iclr.cc/virtual/2021/oral/3458) &nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;[MiniImageNet](https://image-net.org/download.php) &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp;&nbsp;<br>&nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp;&nbsp; &nbsp;[TinyImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip) &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;| &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;[Demo](scripts/models/ResNet.sh) &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;|


## 4 Supported Long-tail methods
### 4.1 Method introduction
- **[Square root resampling](https://openaccess.thecvf.com/content_ECCV_2018/papers/Dhruv_Mahajan_Exploring_the_Limits_ECCV_2018_paper.pdf)**: This paper is from ECCV(2018). It uses public datasets to update the distillation model during aggregation. You can find the method description [here](Baselines/FedMD)
- **[CMO](https://arxiv.org/abs/2112.00412)**: This paper is from CVPR(2022). It uses public datasets to update the distillation model during aggregation. You can find the method description [here](Baselines/FedMD)
- **[Dynamic curriculum learning](https://arxiv.org/abs/1901.06783)**: This paper is from ICCV(2019). It uses public datasets to update the distillation model during aggregation. You can find the method description [here](Baselines/FedMD)
- **[Remix](https://arxiv.org/abs/2007.03943)**: This paper is from ECCV(2020). It uses public datasets to update the distillation model during aggregation. You can find the method description [here](Baselines/FedMD)
- **[GLMC](https://arxiv.org/abs/2305.08661)**: This paper is from CVPR(2023). It uses public datasets to update the distillation model during aggregation. You can find the method description [here](Baselines/FedMD)
- **[OTmix](https://proceedings.neurips.cc/paper_files/paper/2023/hash/bdabb5d4262bcfb6a1d529d690a6c82b-Abstract-Conference.html)**: This paper is from NIPS(2023). It uses public datasets to update the distillation model during aggregation. You can find the method description [here](Baselines/FedMD)
