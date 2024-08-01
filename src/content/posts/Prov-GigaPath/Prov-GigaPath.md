---
title: Prov-GigaPath
published: 2024-07-31
description: ''
image: ''
tags: [机器学习,预训练模型]
category: '论文阅读'
draft: false 
---

## Abstract

我们训练了Prob-GigaPath,在来自171186张WSI包含28中癌症的1.3billion张256x256大小的切片上进行了预训练,这些数据包含了超过3w个病人,我们的数据在26个任务上25个都取了SOTA,其中18个遥遥领先.==有钱就是吊==

## Introduction

虽然当前的任务取得了不错的结果但是需要很多标记数据来进行自监督学习.但是数据标注很贵而且费时间,最近,自监督学习的pretrain模型在无标签任务上展现出了很好的效果.

当前有三种挑战阻碍了病理学基础模型的应用.

1. 公开数据集很稀少并且质量不一.当前的预训练模型大都在TCGA上进行预训练,虽然数据很多,但是还不够多,在不同分布的任务上效果大幅度下降.
2. 现有的方法经常把每个tile当做一个独立的小块,然后使用MIL来得到大块的特征,这种方式限制了模型捕捉复杂全局信息的能力.一个值得注意的方法就是Hierarchical Image Pyramid Transformer (HIPT)分层图像金字塔,能够在tile之间进行attention机制.
3. 还是数据太少

因此我们研发了Prov-GigaPath来解决上述的3种问题,

1. 我们的模型在Prov-Path上进行预训练,训练数据比TCGA大5倍,我们的训练数据是最多的
2. 为了同时捕捉局部和全局的特征,我们提出了GigaPath,一种全局的vit来训练,在很大的图片上进行训练.核心思想就是把image_tiles当做visual token,然后把一个slide变成一个很长的sequence of token.虽然Transformer很强,但是我们不能直接用在这么大的图片上,为了解决这个问题，我们利用了扩张自注意力机制，适应了我们最近开发的LongNet方法。预训练从使用标准视觉变换器的DINOv2进行图像级自监督学习开始，然后通过使用LongNet的掩码自编码器进行全幻灯片级自监督学习。
3. 最后,为了加速研究病理的人的研究,我们把我们的代码开源了,还有模型

## overview of prov-gigapath

我们的模型吧image tile作为输入然后输出slide级别的embedding来作为后续任务的阿巴阿巴.模型包含一个tile encoder来捕捉局部特征,并且有一个slide encoder来得到全局特征.具体细节为:首先把每个tile都转为tile embedding,然后slide encoder输入embedding序列并且生成真个上下文的嵌入.tile encoder通过DINOv2来训练,这是一个自监督学习的框架,slide encoder使用masked 自动编码器来训练.在下游任务上,slide encoder的输出被softmax attention层聚类.我们的模型可以根据下游任务进行各种fine-tune

![模型总览](https://p.ipic.vip/5thlpg.png)

## preprocessing WSI

我们把WSI调整为0.5um每像素(mpp),也就是20倍放大,然后裁剪256x256大小的图片,最终得到了13亿个tile.

微调的时候我们冻结了tile $$encoder然后只微调slide encoder,

## how to 调包

