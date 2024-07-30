---
title: OpenGan
published: 2024-07-29
description: ''
image: ''
tags: [机器学习,生成网路]
category: '论文阅读'
draft: false 
---

## Abstract

真实世界的机器学习系统需要分析与训练数据不同的测试集,在k类别分类中,这被明确的表示为开放集识别,意义在于识别k个类别以外的数据.有两个概念上优雅的开放集识别方式为:

1) 通过利用一些异常数据作为开放集，分辨学习一个开放与闭集的二元判别器；
2) 使用生成对抗网络（GAN）无监督学习闭集数据分布，将其判别器作为开放集似然函数。

但是前面的方法因为对训练数据集过拟合而效果很差,后者也不好,因为GAN的不稳定训练.根据上述信息,我们发明了OpenGAN,结合了两种方法,并且把他们的优点结合.

## 介绍

在机器学习系统中总能遇到在训练集中没有的见过的样本.在K-way分类中,这种任务可以被清晰地定义为开放集识别,需要把开放集的内容识别为k+1的类.一种处理的方式是使用GAN的鉴别器来认识训练数据的分布,但是因为训练的不稳定性导致效果较差.

因此我们开发了opengan,一种逐渐提升开放集准确率的方法,我们展示了使用离群点数据来训练一个鉴别器打倒了SOTA的效果.第二,通过离群值暴露，我们通过对抗性生成欺骗二元判别器的假开放示例来扩充可用的开放训练数据集（图1c）。第三且最重要的是，我们不是在像素上定义判别器，而是在现有的网络的==生成特征==上定义它们（图1d）。我们发现这样的判别器具有更好的泛化能力。

## 方法

在她的代码中,她直接把resnet18的分类层前的代码直接输入了,非常的巧妙,他把512维度的Embedding转换为了512x1x1大小的类似图片的形状后再接着进行输入,通过这种方法,我们就可以使用二维卷积了,不需要自己创造一个GAN网络结构

```python
class Generator(nn.Module):
    def __init__(self, ngpu=1, nz=100, ngf=64, nc=512):
      	# 输入一个噪声噪声(batch_size,nz,1,1),输出生成的特征(batch_size,nc,1,1),使用卷积的方式进行
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.Conv2d( self.nz, self.ngf * 8, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            nn.Conv2d(self.ngf * 8, self.ngf * 4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            nn.Conv2d( self.ngf * 4, self.ngf * 2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            nn.Conv2d( self.ngf * 2, self.ngf*4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf*4),
            nn.ReLU(True),
            nn.Conv2d( self.ngf*4, self.nc, 1, 1, 0, bias=True),
        )

    def forward(self, input):
        return self.main(input)

    
class Discriminator(nn.Module):
    def __init__(self, ngpu=1, nc=512, ndf=64):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.nc = nc
        self.ndf = ndf
        self.main = nn.Sequential(
            nn.Conv2d(self.nc, self.ndf*8, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf*8, self.ndf*4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf*4, self.ndf*2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf*2, self.ndf, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf, 1, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )
		# 最终输出一个(batch_size,1,1,1)的概率,用于判别是不是真的
    def forward(self, input):
        return self.main(input)
```



