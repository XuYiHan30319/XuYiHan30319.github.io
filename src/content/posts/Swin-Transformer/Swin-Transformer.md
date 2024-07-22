---
title: Swin-Transformer
published: 2024-07-22
description: ""
image: ""
tags: [机器学习,CVAt]
category: "论文阅读"
draft: false
---

::github{repo="microsoft/Swin-Transformer"}

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

SwinT在分类的时候直接平均输出分类.而ViT是专门做一个可学习参数cls.并且SwinT的位置编码是可选的.

```python
class SwinTransformer(nn.Module):
    def __init__(...):
        super().__init__()
        ...
        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            
        self.pos_drop = nn.Dropout(p=drop_rate)

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(...)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
```

### Patch EMbedding

首先要把图片切成一块一块的,然后嵌入向量.具体做法是把原始图片裁剪为一个个`pathc_size x patchsize`的窗口大小然后再嵌入.

```python
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size) # -> (img_size, img_size)
        patch_size = to_2tuple(patch_size) # -> (patch_size, patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # 假设采取默认参数
        x = self.proj(x) # 出来的是(N, 96, 224/4, 224/4) 
        x = torch.flatten(x, 2) # 把HW维展开，(N, 96, 56*56)
        x = torch.transpose(x, 1, 2)  # 把通道维放到最后 (N, 56*56, 96)
        if self.norm is not None:
            x = self.norm(x)
        return x
```

通过一个卷积操作就完成了patch并且嵌入向量了

### Patch Merging

该模块的作用是在每个Stage开始前做降采样，用于缩小分辨率，调整通道数进而形成层次化的设计，同时也能节省一定运算量。每次降采样是2倍,在行和列方向上间隔2选取元素,然后拼接在一起成为一整个张量,最后展开,此时HW各缩小2倍,通道数变为4倍,子啊通过全连接层把通道数变为2倍

```python
class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x
```

![image-20240722140645525](https://p.ipic.vip/upfv1t.png)

### Window Partition/Reverse

```python
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x
```

### Window Attention

传统的Transformer都是基于全局来计算注意力的,而Swin Transformer把注意力限制在每个窗口内,减少计算量.

:::tip

看不懂,跳了

:::

### Shifted Window Attention

为了更好的跟其他的window进行交互,SwinT还引入了shifted window

![image-20240722141541448](https://p.ipic.vip/ezj7mr.png)

左边是没有重叠的Window Attention，而右边则是将窗口进行移位的Shift Window Attention。可以看到移位后的窗口包含了原本相邻窗口的元素。但这也引入了一个新问题，即**window的个数翻倍了**，由原本四个窗口变成了9个窗口。

在代码中,通过特征图唯一,并给Attention设置了mask来间接实现的,能够在保持原有window的个数下,最后的计算结果等价.

![image-20240722141656749](https://p.ipic.vip/e65yx5.png)

:::tip

不想看,跳了

:::

### Transformer Block整体架构

![image-20240722141816900](https://p.ipic.vip/vwp58q.png)

两个连续的blocak如上所示,注意一个stage包含的block个数必须是偶数,因为需要包含的交替`window attention`和`Shifted Window Attention`.

