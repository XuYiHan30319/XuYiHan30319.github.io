---
title: Swin-Unet[R]
published: 2024-07-22
description: ''
image: ''
tags: [机器学习,CV,分割]
category: '论文阅读'
draft: true
---

## Abstract

在近几年,U形状网络和跳跃链接在医学任务上取得了巨大的成就.但是CNN网络因为卷积的局部特性无法学习到全局和大范围的分割信息.本文中使用层级Swin-transformer来作为encoder提取特征.并且设计了一种基于transformer的上采样方式.

在传统的UNet中,使用一系列的卷积和下采样方式来得到将大范围和感受野.然后再使用上采样恢复pix2pix的特征.通过这种形式,出现了很多变体,比如Res-Unet,Dense-Unet,U-Net++..并且进化到三维领域比如3D-Unet和V-Net.

Transformer是从NLP领域发展出来的,并且由ViT第一次引入视觉领域.和CNN相比,ViT的缺点是需要再大规模数据集上进行pre-train.为了减轻ViT的训练困难,有一些训练策略被提出了.值得注意的是其中有一个叫做SwinTransformer的架构被作为视觉方向的backbone.通过滑动窗口策略,SwinTransformer在分类,目标检测和语义分割上都取得了SOTA的成果.在这个任务重,我们尝试使用SwinTransformer作为基本单元来创建一个U形状Encoder-Decoder架构的网络来帮助图像分割.

UNETR是第一个完全利用ViT作为encoder而不依赖CNN的特诊提取器.

## 具体方法

![image-20240722103033888](https://p.ipic.vip/pxp85u.png)

输入的大小是3D的多模态MRI图像,有4通道.Swin UNETER创建没有重叠的patched然后计算自注意力机制.计算出来的特征被输入到CNN-Decoder中,最终的输出是HxWxDx3大小的3分类分割.
