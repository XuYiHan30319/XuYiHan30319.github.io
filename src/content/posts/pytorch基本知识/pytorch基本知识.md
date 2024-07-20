---
title: pytorch基本知识
published: 2024-07-16
description: ''
image: ''
tags: [机器学习]
category: ''
draft: true 
---

## conv1d

```python
torch.nn.Conv1d(in_channels,       "输入图像中的通道数"
                out_channels,      "卷积产生的通道数"
                kernel_size,       "卷积核的大小"
                stride,            "卷积的步幅。默认值：1"
                padding,           "添加到输入两侧的填充。默认值：0"
                dilation,          "内核元素之间的间距。默认值：1"
                groups,            "从输入通道到输出通道的阻塞连接数。默认值：1"
                bias,              "If True，向输出添加可学习的偏差。默认：True"
                padding_mode       "'zeros', 'reflect', 'replicate' 或 'circular'. 默认：'zeros'"
                )
```

### 输入输出的大小变化

input – （批大小， 数据的通道数， 数据长度）
output –（批大小， 产生的通道数， 卷积后长度）

卷积后的维度：(n - k + 2 * p ) / s + 1
k: 卷积核大小，p: 使用边界填充，s: 步长。

![image-20240718132130898](https://p.ipic.vip/1rwxiy.png)

## Batch Normalization

Batch Normalization(批标准化),和普通的数据标准化类似,是把分散的数据统一的一种做法,也是优化神经网络的一种方法,具有统一规格的数据, 能让机器学习更容易学习到数据之中的规律.Batch normalization 也可以被看做一个层面. 在一层层的添加神经网络的时候, 我们先有数据 X, 再添加全连接层, 全连接层的计算结果会经过 激励函数 成为下一层的输入, 接着重复之前的操作. Batch Normalization (BN) 就被添加在每一个全连接和激励函数之间.

计算结果在进入激活函数前的值非常重要,也就是数据的分布对与激活函数来说很重要.大部分数据在一个去建立才能有效的进行传递.

## Dropout暂退法

 
