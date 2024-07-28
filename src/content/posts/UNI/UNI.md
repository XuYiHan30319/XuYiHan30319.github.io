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

在我们的工作中我们映入了一个病理的vision encoder,他是一个general-purpose,self-supervised的模型,UNI,是一个ViT-L在其中一个最大的病例数据集(Mass-100k)上进行预训练的模型.在预训练阶段,我们使用DINOv2的自监督学习方法
