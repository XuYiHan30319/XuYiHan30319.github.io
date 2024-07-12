---
title: SimCLR2
published: 2024-07-13
description: ''
image: ''
tags: [机器学习,分类,自监督学习,CV]
category: '论文阅读'
draft: false 
---
::github{repo="google-research/simclr"}

[Self-Supervised Learning 超详细解读 (二)：SimCLR系列 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/378953015)

## 前言

> SimCLR v2相对SimCLR做了什么改进?

**答**：SimCLR v2 的第 1 个发现是：

在使用无标签数据集做 Pre-train 的这一步中，模型的尺寸很重要，用 **deep and wide**的模型可以帮助提升性能。

SimCLR v2 的第 2 个发现是：

使用无标签数据集做 Pre-train 完以后，现在要拿着有标签的数据集 Fine-tune 了。之后再把这个 **deep and wide 的模型** 蒸馏成一个更小的网络。

所以，SimCLR v2 的方法，用8个英文单词概括一下就是：Unsupervised Pre-train, Supervised Fine-tune，Distillation Using Unlabeled Data.

**所以 SimCLR v2 论文里面给出了3个论点：**

- 对于半监督学习来讲，**在标签量极少的情况下，模型越大，获益就越多**。这很不符合直觉，常识是标签这么少了，模型变大会过拟合。
- 即使模型越大能够学到越 general 的 representations，但是这是在不涉及下游任务的task-agnostic 的情况下。**一旦确定了下游任务，就不再需要大模型了，可以蒸馏成一个小模型。**
- **Projection head 很重要**，更深的 Projection head 可以学习到更好的representation，在下游任务做 Fine-tune 之后也更好。

## 具体步骤

![img](https://pic4.zhimg.com/80/v2-669d2a660f5885d784d46d35e2e84457_720w.webp)

SimCLR v2的具体步骤可以分为以下3步：

1. **Unsupervised Pre-train：**使用**无标签数据**以一种 **Task-agnostic** 的方式**预训练**Encoder，得到比较 general 的 Representations。
2. **Supervised Fine-tune：**使用**有标签数据**以一种 **Task-specific** 的方式 **Fine-tune** Encoder。
3. **Distillation Using Unlabeled Data：**使用**无标签数据**以一种 **Task-specific** 的方式**蒸馏** Encoder，得到更小的Encoder。

第一步其实跟SIMCLR一样的,区别在于

1. 把Encoder换成了ResNet152和selective kernels.
2. projection head变深,原来是两个FC+一个relu,现在是3个FC
3. 加入MoCo内存机制

第二部的Fine-tune中保留了Projection head,保留了一半扔掉一般做Fine-tune.

第三部中把Fine-tune后的网络作为Teacher去蒸馏一个更小的student网络.下面是只使用蒸馏而不用任何label的情况，当我们也有一些label的时候，也可以在无监督的损失函数的基础上添加一项有监督的损失,具体的损失函数不写出来了：

![img](https://pic3.zhimg.com/80/v2-5d7be4a15e1354c1018177d69bb86eba_720w.webp)

## 总结

之前的使用自监督获得representations需要特定的算法。大牛上来就说之前的方法多数属于**生成方法或者判别方法。**告诉我们**contrastive learning**才是目前的宠儿。过去一年来，相继有CPC, CMC, MoCo 出来，想要解决的共同一个问题就是，如何提高softmax中negatives的数量。其中 CPC 用了patch based方法，CMC 用了一个memory buffer，MoCo 用了momentum update去keep一个negative sample queue。

这篇文章告诉大家，只要机子够多，batch size够大，每个batch中除了positive以外的都当negatives就已经足够了。
