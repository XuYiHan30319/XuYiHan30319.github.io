---
title: ViT
published: 2024-07-22
description: ''
image: ''
tags: [机器学习,CV]
category: '论文阅读'
draft: false 
---

## 简介

ViT是Google提出的把Transformer应用在图像分类的模型.因为模型简单效果好可扩展性强,所以成为了Transformer在CV领域的里程碑

ViT论文最核心的结论就是当有足够多的数据进行预训练,ViT就会超过CNN,突破Transformer没有归纳偏置的限制,可以在下游任务重获得较好的迁移学习效果.

而数据量少的时候,ViT比同等大小的ResNet小一点,因为Transformer缺少归纳偏置(一种先验知识,是CNN预先设计好的假设).CNN有两种归纳偏置:其一是局部性,即在图片上相邻的区域具有相似特征,一种是平移不变形,$f(g(x))=g(f(x))$,其中f是平移,g是卷积,当CNN有了上面的归纳偏置,就有了很多的先验知识,需要相对少的数据就可以学习一个好模型.

## ViT结构

![ViT架构](https://p.ipic.vip/x1mzr1.png)

ViT首先把图片划分为多个Patch(16*16),然后把每个Patch投影为固定长度的向量送入Transformer中.后续encoder的操作就和原始Transformer一模一样.但是因为对图片分类，因此在输入序列中加入一个特殊的token，该token对应的输出即为最后的类别预测.

一个ViT block如下

1. patch embedding:比如输入图片大小为224x224,把图片分为固定大小的patch,patch大小为16x16,每个图片就变成了196个patch,即输入序列长度为196,每个patch的维度是16x16x3=768个,线性投影层的维度是768x768,所以每个patch出来还是768,即具有196个token,每个token维度768.最终维度是197x768,到目前为止,已经把视觉问题变成了seq2seq问题.
2. positional encoding:vit需要加入位置编码,可以理解为一张表,一共N行,N的大小和输入序列长度相同,每一行代表一个向量.每一行代表一个向量，向量的维度和输入序列embedding的维度相同（768）。注意位置编码的操作是sum，而不是concat。加入位置编码信息之后，维度依然是**197x768**
3. 多头注意力机制:LN输出维度依然是197x768。多头自注意力时，先将输入映射到q，k，v，如果只有一个头，qkv的维度都是197x768，如果有12个头（768/12=64），则qkv的维度是197x64，一共有12组qkv，最后再将12组qkv的输出拼接起来，输出维度是197x768，然后在过一层LN，维度依然是**197x768**
4. MLP：将维度放大再缩小回去，197x768放大为197x3072，再缩小变为**197x768**

这样就是一个标准block了,输入和输出大小都一样,因此可以堆叠多个block.对于分类,是这样的:

```python
import torch
import torch.nn as nn

class ViTClassifier(nn.Module):
    def __init__(self, num_classes, hidden_dim):
        super(ViTClassifier, self).__init__()
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # 假设 x 的形状是 [batch_size, num_patches + 1, hidden_dim]
        cls_token_output = x[:, 0]  # 取出 [CLS] token 的输出 只要每个的第一个
        logits = self.classifier(cls_token_output)  # 线性层映射到类别数
        return logits

# 示例用法
num_classes = 10
hidden_dim = 768
model = ViTClassifier(num_classes, hidden_dim)

# 假设输入张量的形状是 [batch_size, num_patches + 1, hidden_dim]
input_tensor = torch.randn(32, 197, 768)  # batch_size=32
output = model(input_tensor)
print(output.shape)  # 应该是 [32, num_classes]
```



实际开发中的做法是：基于大数据集上训练，得到一个预训练权重，然后再在小数据集上Fine-Tune。
