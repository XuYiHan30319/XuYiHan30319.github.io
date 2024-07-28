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

## Normalization

Batch Normalization(批标准化),和普通的数据标准化类似,是把分散的数据统一的一种做法,也是优化神经网络的一种方法,具有统一规格的数据, 能让机器学习更容易学习到数据之中的规律.Batch normalization 也可以被看做一个层面. 在一层层的添加神经网络的时候, 我们先有数据 X, 再添加全连接层, 全连接层的计算结果会经过 激励函数 成为下一层的输入, 接着重复之前的操作. Batch Normalization (BN) 就被添加在每一个全连接和激励函数之间.

计算结果在进入激活函数前的值非常重要,也就是数据的分布对与激活函数来说很重要.大部分数据在一个去建立才能有效的进行传递.

三维医学图像处理中,现存不足时经常遇到的问题,模型应该在batch size和patch size之间做出权衡.Unet中应该优先考虑patch_size,保证模型能获得足够的信息来进行推理,但是batch size的最小值应该大于等于2,因为我们需要保证训练过程中优化的鲁棒性.在保证patch size的情况下如果现存有多余,再增加batch size.因为batch size都比较小,所以大多使用Instance Norm而不是BN.

- BatchNorm：batch方向做归一化，算NxHxW的均值，对小batchsize效果不好；BN主要缺点是对batchsize的大小比较敏感，由于每次计算均值和方差是在一个batch上，所以如果batchsize太小，则计算的均值、方差不足以代表整个数据分布。
- LayerNorm：channel方向做归一化，算CxHxW的均值，主要对RNN(处理序列)作用明显，目前大火的Transformer也是使用的这种归一化操作；
- InstanceNorm：一个channel内做归一化，算H*W的均值，用在风格化迁移；因为在图像风格化中，生成结果主要依赖于某个图像实例，所以对整个batch归一化不适合图像风格化中，因而对HW做归一化。可以加速模型收敛，并且保持每个图像实例之间的独立。
- GroupNorm：将channel方向分group，然后每个group内做归一化，算(C//G)HW的均值；这样与batchsize无关，不受其约束，在分割与检测领域作用较好。

## Data Parallel

pytorch官方的数据并行类是

```python
torch.nn.DataParallel(model,device_ids=None,output_device=None,dim=0)
```

主要把input数据按照batch这个维度,把数据划分到指定的设备上.其他的对象复制到每个设备上.在向前传播的过程中,model被复制到每个设备上,每个复制的副本处理一部分数据数据,在反向传播的过程中,每个副本的module的梯度都被汇聚到原始的model上进行计算.

:::important

Batch size的大小一定要大于GPU的数量,实践中batch size的大小一般设置为GPU的倍数

:::

device_ids是所有可操作的GPU号,output_device是输出汇总到的指定GPU,默认为device_ids[0]号

## nn.Liner
在 PyTorch 中，nn.Linear 层的工作原理是对最后一个维度进行线性变换。换句话说，无论输入张量有多少个维度，nn.Linear 层只会对最后一个维度进行线性变换，而不会影响其他维度。因此，如果输入张量的形状是 (batch, n, dim)，那么 nn.Linear 层会对 dim 维度进行变换，而保持 batch 和 n 维度不变。

总之,liner层只会对最后一个维度线性变换.

## 混合精度训练

日常的使用通常使用单精度浮点表示(float32),单精度和半精度的表示如下,与双精度表示,半精度仅有16bit

![image-20240728162624237](https://p.ipic.vip/qisxcy.png)

所谓混合精度训练,就是单精度和半精度混合,float16和float相比内存少,计算快

- 内存少:只需要一般的精度,memory-bandwidht减半,模型的batch size可以更大,训练的时候多卡的交互(通信量)减少,减少等待时间,加快数据沟通
- 计算快:GPU针对16fp进行优化,吞吐量可以达到单精度的2~8倍

那为什么要用混合精度呢?

- 数据溢出Underflow:fp16的范围下载,大概$2^{-24}到65504$之间,对于深度学习来说,最大的问题在于后期的梯度计算的时候会发生梯度消失
- 舍入误差:fp16weight:$2^{-3}$,gradient:$2^{-14}$,那么这俩加起来还是$2^{-3}$,这就是舍入误差.

为了解决这个问题,我们的方法如下:

- fp32备份权重,主要解决舍入误差,可以概括为：weights, activations, gradients 等数据在训练中都利用FP16来存储，同时拷贝一份FP32的weights，用于更新,确保在更新的时候是在float32下进行的
- loss scale:主要解决underflow问题,由于链式法则的存在,loss上的scale会作用在梯度上.只有在进行更新的时候，才会将 scaled-gradient 转化为 fp32，同时将scale抹去。
- 提高算数精度:在某些模型中，fp16矩阵乘法的过程中，需要利用 fp32 来进行矩阵乘法中间的累加(accumulated)，然后再将 fp32 的值转化为 fp16 进行存储。 换句不太严谨的话来说，也就是利用 **利用fp16进行乘法和存储，利用fp32来进行加法计算**。 这么做的原因主要是为了减少加法过程中的舍入误差，保证精度不损失。因此只有特定gpu才可以这么做