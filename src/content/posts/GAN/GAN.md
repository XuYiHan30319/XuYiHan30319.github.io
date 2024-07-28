---
title: GAN
published: 2024-07-24
description: ''
image: ''
tags: [机器学习]
category: '论文阅读'
draft: false 
---

## 对抗生成网络(GANs)综述

所谓生成模型,就是指可以描述成一个生成数据的模型,属于一种概率模型,生成模型能够随机生成观测数据的模型,尤其是在给定某些隐含参数的情况下,他给观测值和标注数据序列一个联合概率分布.在机器学习中,生成模型可以直接用来对数据建模,也可以简历变量间的联合概率分布.通俗来说就是生成不包含在训练集中的新数据.

我们常见的模型一般属于判别模型,判别模型可以简单的理解为分类.因此生成模型和判别模型的区别在于

1. 生成模型的数据集是没有和判别模型类似的标签的,是一种非监督学习.
2. 判别模型:p(y|x)表示给定观测x得到y的概率,生成模型p(x) 即观测x出现的概率。如果有标签则表示为: p(x|y) 指定标签y生成x的概率。

 而GAN模型的诞生，就是结合了生成模型的特点与判别模型的特点，通过动态对抗的方式进行训练，在同态平衡中寻找最优解。

## 什么是GAN

### 对抗思想

假设一个城市治安混乱，很快，这个城市里就会出现无数的小偷。在这些小偷中，有的可能是盗窃高手，有的可能毫无技术可言。假如这个城市开始整饬其治安，突然开展一场打击犯罪的「运动」，警察们开始恢复城市中的巡逻，很快，一批「学艺不精」的小偷就被捉住了。之所以捉住的是那些没有技术含量的小偷，是因为警察们的技术也不行了，在捉住一批低端小偷后，城市的治安水平变得怎样倒还不好说，但很明显，城市里小偷们的平均水平已经大大提高了。警察们开始继续训练自己的破案技术，开始抓住那些越来越狡猾的小偷。随着这些职业惯犯们的落网，警察们也练就了特别的本事，他们能很快能从一群人中发现可疑人员，于是上前盘查，并最终逮捕嫌犯；小偷们的日子也不好过了，因为警察们的水平大大提高，如果还想以前那样表现得鬼鬼祟祟，那么很快就会被警察捉住。为了避免被捕，小偷们努力表现得不那么「可疑」，而魔高一尺、道高一丈，警察也在不断提高自己的水平，争取将小偷和无辜的普通群众区分开。随着警察和小偷之间的这种「交流」与「切磋」，小偷们都变得非常谨慎，他们有着极高的偷窃技巧，表现得跟普通群众一模一样，而警察们都练就了「火眼金睛」，一旦发现可疑人员，就能马上发现并及时控制——最终，我们同时得到了最强的小偷和最强的警察。

AlphaGo的中间版本使用两个相互竞争的网络。对抗性示例是指与真实示例非常不同，但被非常自信地归入真实类别的示例，或与真实示例略有不同，但被归入错误类别的示例。这是最近一个非常热门的研究课题。对抗学习是一个极大极小值的问题.防卫者购机拿了我们想要的争取的分类器,同时,攻击者探索模型的输入让成本最大化.

### GAN(generative Adversarial Network)

gan是一个生成和对抗并存的网路,包含一个生成器和一个判别器,生成器来根据需要生成越来越接近真实标签的数据,判别器用来不断区分生成器生成的结果和标签之间的区别.

和其他生成算法相比，GANs的提出是为了克服其他生成算法的缺点。对抗式学习背后的基本思想是，生成器试图创建尽可能真实的示例来欺骗鉴别器。鉴别器试图区分假例子和真例子。生成器和鉴别器都通过对抗式学习进行改进。这种对抗性的过程使GANs比其他生成算法具有显著的优势。

![img](https://ai-studio-static-online.cdn.bcebos.com/91ca9062c8ae487985a34a5253274910ad742420d61d48a98ae207bde7dbdd16)

比如下图中,generator生成图片来尝试欺骗判别器,判别器尽可能分辨出来.对于判别器来说就是一个二分类任务,用最小交叉熵损失函数就好了.实际训练的时候,我们采用交替训练,首先训练D,然后训练G,不断往复,通常迭代k次判别器,然后迭代一次生成器.目标是在判别器预测概率为1/2的时候效果达到最好,这时候分辨不出来谁是谁生成的了.

模型代码如下所示:

```python
import argparse
import os
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch

## 创建文件夹
os.makedirs("./images/gan/", exist_ok=True)         ## 记录训练过程的图片效果
os.makedirs("./save/gan/", exist_ok=True)           ## 训练完成时模型保存的位置
os.makedirs("./datasets/mnist", exist_ok=True)      ## 下载数据集存放的位置

## 超参数配置
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=500, help="interval betwen image samples")
opt = parser.parse_args()
## opt = parser.parse_args(args=[])                 ## 在colab中运行时，换为此行
print(opt)

## 图像的尺寸:(1， 28， 28),  和图像的像素面积:(784)
img_shape = (opt.channels, opt.img_size, opt.img_size)
img_area = np.prod(img_shape)

## 设置cuda:(cuda:0)
cuda = True if torch.cuda.is_available() else False

## mnist数据集下载
mnist = datasets.MNIST(
    root='./datasets/', train=True, download=True, transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ), 
)

## 配置数据到加载器
dataloader = DataLoader(
    mnist,
    batch_size=opt.batch_size,
    shuffle=True,
)


## ##### 定义判别器 Discriminator ######
## 将图片28x28展开成784，然后通过多层感知器，中间经过斜率设置为0.2的LeakyReLU激活函数，
## 最后接sigmoid激活函数得到一个0到1之间的概率进行二分类
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_area, 512),                   ## 输入特征数为784，输出为512
            nn.LeakyReLU(0.2, inplace=True),            ## 进行非线性映射
            nn.Linear(512, 256),                        ## 输入特征数为512，输出为256
            nn.LeakyReLU(0.2, inplace=True),            ## 进行非线性映射
            nn.Linear(256, 1),                          ## 输入特征数为256，输出为1
            nn.Sigmoid(),                               ## sigmoid是一个激活函数，二分类问题中可将实数映射到[0, 1],作为概率值, 多分类用softmax函数
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)            ## 鉴别器输入是一个被view展开的(784)的一维图像:(64, 784)
        validity = self.model(img_flat)                 ## 通过鉴别器网络
        return validity                                 ## 鉴别器返回的是一个[0, 1]间的概率

      
## ###### 定义生成器 Generator #####
## 输入一个100维的0～1之间的高斯分布，然后通过第一层线性变换将其映射到256维,
## 然后通过LeakyReLU激活函数，接着进行一个线性变换，再经过一个LeakyReLU激活函数，
## 然后经过线性变换将其变成784维，最后经过Tanh激活函数是希望生成的假的图片数据分布, 能够在-1～1之间。
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        ## 模型中间块儿
        def block(in_feat, out_feat, normalize=True):           ## block(in， out )
            layers = [nn.Linear(in_feat, out_feat)]             ## 线性变换将输入映射到out维
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))    ## 正则化
            layers.append(nn.LeakyReLU(0.2, inplace=True))      ## 非线性激活函数
            return layers
        ## prod():返回给定轴上的数组元素的乘积:1*28*28=784
        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),       ## 线性变化将输入映射 100 to 128, 正则化, LeakyReLU
            *block(128, 256),                                   ## 线性变化将输入映射 128 to 256, 正则化, LeakyReLU
            *block(256, 512),                                   ## 线性变化将输入映射 256 to 512, 正则化, LeakyReLU
            *block(512, 1024),                                  ## 线性变化将输入映射 512 to 1024, 正则化, LeakyReLU
            nn.Linear(1024, img_area),                          ## 线性变化将输入映射 1024 to 784
            nn.Tanh()                                           ## 将(784)的数据每一个都映射到[-1, 1]之间
        )
    ## view():相当于numpy中的reshape，重新定义矩阵的形状:这里是reshape(64, 1, 28, 28)
    def forward(self, z):                                       ## 输入的是(64， 100)的噪声数据
        imgs = self.model(z)                                     ## 噪声数据通过生成器模型
        imgs = imgs.view(imgs.size(0), *img_shape)                 ## reshape成(64, 1, 28, 28)
        return imgs                                              ## 输出为64张大小为(1, 28, 28)的图像


## 创建生成器，判别器对象
generator = Generator()
discriminator = Discriminator()

## 首先需要定义loss的度量方式  （二分类的交叉熵）
criterion = torch.nn.BCELoss()

## 其次定义 优化函数,优化函数的学习率为0.0003
## betas:用于计算梯度以及梯度平方的运行平均值的系数
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

## 如果有显卡，都在cuda模式中运行
if torch.cuda.is_available():
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion = criterion.cuda()



## ----------
##  Training
## ----------
## 进行多个epoch的训练
for epoch in range(opt.n_epochs):                               ## epoch:50
    for i, (imgs, _) in enumerate(dataloader):                  ## imgs:(64, 1, 28, 28)     _:label(64)
        
        ## =============================训练判别器==================
        ## view(): 相当于numpy中的reshape，重新定义矩阵的形状, 相当于reshape(128，784)  原来是(128, 1, 28, 28)
        imgs = imgs.view(imgs.size(0), -1)                          ## 将图片展开为28*28=784  imgs:(64, 784)
        real_img = Variable(imgs).cuda()                            ## 将tensor变成Variable放入计算图中，tensor变成variable之后才能进行反向传播求梯度
        real_label = Variable(torch.ones(imgs.size(0), 1)).cuda()      ## 定义真实的图片label为1
        fake_label = Variable(torch.zeros(imgs.size(0), 1)).cuda()     ## 定义假的图片的label为0


        ## ---------------------
        ##  Train Discriminator
        ## 分为两部分：1、真的图像判别为真；2、假的图像判别为假
        ## ---------------------
        ## 计算真实图片的损失
        real_out = discriminator(real_img)                          ## 将真实图片放入判别器中
        loss_real_D = criterion(real_out, real_label)               ## 得到真实图片的loss
        real_scores = real_out                                      ## 得到真实图片的判别值，输出的值越接近1越好
        ## 计算假的图片的损失
        ## detach(): 从当前计算图中分离下来避免梯度传到G，因为G不用更新
        z = Variable(torch.randn(imgs.size(0), opt.latent_dim)).cuda()      ## 随机生成一些噪声, 大小为(128, 100)
        fake_img = generator(z).detach()                                    ## 随机噪声放入生成网络中，生成一张假的图片。 
        fake_out = discriminator(fake_img)                                  ## 判别器判断假的图片
        loss_fake_D = criterion(fake_out, fake_label)                       ## 得到假的图片的loss
        fake_scores = fake_out                                              ## 得到假图片的判别值，对于判别器来说，假图片的损失越接近0越好
        ## 损失函数和优化
        loss_D = loss_real_D + loss_fake_D                  ## 损失包括判真损失和判假损失
        optimizer_D.zero_grad()                             ## 在反向传播之前，先将梯度归0
        loss_D.backward()                                   ## 将误差反向传播
        optimizer_D.step()                                  ## 更新参数


        ## -----------------
        ##  Train Generator
        ## 原理：目的是希望生成的假的图片被判别器判断为真的图片，
        ## 在此过程中，将判别器固定，将假的图片传入判别器的结果与真实的label对应，
        ## 反向传播更新的参数是生成网络里面的参数，
        ## 这样可以通过更新生成网络里面的参数，来训练网络，使得生成的图片让判别器以为是真的, 这样就达到了对抗的目的
        ## -----------------
        z = Variable(torch.randn(imgs.size(0), opt.latent_dim)).cuda()      ## 得到随机噪声
        fake_img = generator(z)                                             ## 随机噪声输入到生成器中，得到一副假的图片
        output = discriminator(fake_img)                                    ## 经过判别器得到的结果
        ## 损失函数和优化
        loss_G = criterion(output, real_label)                              ## 得到的假的图片与真实的图片的label的loss
        optimizer_G.zero_grad()                                             ## 梯度归0
        loss_G.backward()                                                   ## 进行反向传播
        optimizer_G.step()                                                  ## step()一般用在反向传播后面,用于更新生成网络的参数




        ## 打印训练过程中的日志
        ## item():取出单元素张量的元素值并返回该值，保持原元素类型不变
        if (i + 1) % 100 == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [D real: %f] [D fake: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), loss_D.item(), loss_G.item(), real_scores.data.mean(), fake_scores.data.mean())
            )
        ## 保存训练过程中的图像
        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(fake_img.data[:25], "./images/gan/%d.png" % batches_done, nrow=5, normalize=True)

## 保存模型
torch.save(generator.state_dict(), './save/gan/generator.pth')
torch.save(discriminator.state_dict(), './save/gan/discriminator.pth')
```

## GAN在医学中的应用

大致可以分为

- one-to-oen synthesis methods:从一个模态生成另一个模态
- Many-to-one sythesis:从给定的几个模态中生成目标模态
- unified synthesis:可以通过任意模态生成任意模态

当然了,可以看出来unified synthesis是最好的,但是也存在一些问题,比如一些结构细节看不清楚,这因为:

1.  已知的方法依赖于单个encoder或者一些特定模态的encoder来处理输入,不能很好的结合模态之间的数据,这导致了细节的缺失和模态的不完全融合.
2. 为了确保网络对不同的模态鲁邦,已知的unified方法只是使用了一个最大池化来得到统一的潜在特征.这可能导致模态细节的丢失.

## one-to-one synthesis

一对一的方合成法以一个可用的对比度作为输入并且生成单个目标对比度.早期的方法通常基于 基于patch的回归,稀疏自垫表示法和atlas,这些方法的性能收到手工设计特征的限制.随着CNN的发展,现在多用CNN等深度学习来one-to-one的图像生成.比如用GAN来,有什么使用3D CNN来生成MR到CT图像的映射(pGAN和cGAN),diffusion model当前也成为了图像生成的一种工具.



## PatchGAN

```python
lass NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
```

其实就是把生成图像的每个区域都说明是真的还是假的,注意到模型的最后一层的输出维度为1,也就是这个区域是真的还是假的,作用是让生成的图像没有那么模糊.损失函数用MSE就可以了(也就是L2损失函数均方误差,L1损失就是MAE平均绝对误差)

## ResViT代码结构

首先是损失函数

```python
# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(
        self,
        use_lsgan=True,
        target_real_label=1.0,
        target_fake_label=0.0,
        tensor=torch.FloatTensor,
    ):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    # 创建对应的标签,分别存在self.real_label_var和self.fake_label_var中,如果标签的大小不一致,那么重新创建
    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = (self.real_label_var is None) or (
                self.real_label_var.numel() != input.numel()
            )
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = (self.fake_label_var is None) or (
                self.fake_label_var.numel() != input.numel()
            )
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

```

这个损失函数跟mae差不多,主要是可以自动生成标签~,生成的标签是一个二维的全1或者全0的矩阵

