---
title: Swin-Transformer
published: 2024-07-22
description: ""
image: ""
tags: []
category: ""
draft: false
---



## Abstract

目前Transformer领域在图像上有两大挑战

- 视觉实体变化大,在不同场景下Transformer性能不一定好
- 图像分辨率高,像素多,Transformer基于全局自注意力机制计算量大

针对上述问题,使用了包含**划窗操作,具有层级设计**的Swin Transformer.其中划窗操作包括不重叠的local window,和重叠的cross-window,将注意力限制在一个窗口中,一方面能够引入CNN的局部性又能节省计算量.

![image-20240722131001273](https://p.ipic.vip/9ksj5c.png)

## 整体架构

![image-20240722131046518](https://p.ipic.vip/wkip7g.png)

整个模型采取层级化设计,包含4个stage,每个stage都会缩小输入特征图的分辨率,像CNN一样扩大感受野.

- 开始的时候切成小片,并且嵌入到Embedding中
- 在每个stage中,由patch merging和多个block组成,patch merging模块主要在每个stage开始的时候降低图像分辨率.
- Blocak的具体机制由LN,MLP,Window Attention和shifted window attention组成

SwinT在分类的时候直接平均输出分类.而ViT是专门做一个可学习参数

