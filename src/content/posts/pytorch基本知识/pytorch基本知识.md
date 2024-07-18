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
