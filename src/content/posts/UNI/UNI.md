---
title: UNI
published: 2024-07-28
description: ''
image: ''
tags: [机器学习,病理,预训练]
category: '论文阅读'
draft: false 
---

## Abstract

WSI的高像素和形态的可变性对AI提出了巨大的挑战,目前的方法大多采用自监督学习或者迁移学习的image eocoders,但是没有一种通用的在多种病理上进行训练的模型,所以我们提出了UNI,一种使用自监督学习的通用模型,使用来自20种组织的10000张HE染色的WSI上的超过100million张图片来进行训练

通过自监督迁移学习的模型泛化和缩放(?)能力依赖数据的数量和多样性,在计算机视觉中,很多自监督学习都依赖于ImageNet和其他的大datasets.这些模型因为他们可以适应很多下游任务被称为基模型.在CPath(Contemporary computational pathology)中,TCGA也被很多的自监督学习当做数据来源.但是,当前CPath的预训练模型的数据大小和多样性被限制了.

在我们的工作中我们映入了一个病理的vision encoder,他是一个general-purpose,self-supervised的模型,UNI,是一个ViT-L在其中一个最大的病例数据集(Mass-100k)上进行预训练的模型.在预训练阶段,我们使用DINOv2的自监督学习方法来进行学习.然后再下游进行了各种测试,比如细胞核分割,癌症检测等等

## 结果

基础模型的一个重要特点是它能够在下游任务上展现出优秀的性能提升.UNI模型因为训练数据量大,训练种类多,我们的模型展现除了良好的性能.

作者还研究了UNI在15个slide-level上分类的效果,使用ABMIL比较了从UNI中预提取的特征与其他预训练编码器的特征。在所有15个切片任务重,UNI一致超越了其他所有的预训练编码器,比ResNet-50高出了26.4%,比CTransPath高出了8.3%.在比较的时候,我们发现UNI作为特征提取器的ABMIL超过了许多其他的复杂MIL架构.

但是我们还观察到我们的研究有一定的限制,基于ViT-L架构的UNI缺乏解决CPath中密集预测任务的视觉特定偏差，细胞类型分割性能提升并不像其他任务中观察到的那样显著。

## 怎么调包?

```python
import timm

model = timm.create_model("hf_hub:MahmoodLab/UNI", pretrained=True)
```

上面都是废话,我们会调包就好了,当然,服务器上没有vpn()所以我们需要把模型下载到本地

```python
import timm
import torch
import json
import os

model = timm.create_model(
    "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
)
model.load_state_dict(torch.load(os.path.join("./UNI", "pytorch_model.bin"), map_location="cuda:1"), strict=True)
print(model)
```

