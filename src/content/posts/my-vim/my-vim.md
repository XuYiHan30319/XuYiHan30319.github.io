---
title: my-vim
published: 2024-07-21
description: 'vim的基本设置和neovim配置'
image: ''
tags: []
category: ''
draft: true
---

## vim基本指令

- 移动:hjkl
- 删除:x删除光标所在的位置,使用dw删除一个单词,使用d$删除当前光标位置到行末,包括当前光标
- 插入:按下i后再当前光标的前面进行插入,使用a在当前位置的后面位置进行插入
- 退出:使用:wq保存退出,:q!强制退出
- 撤销:按下u进行撤销
- 复制粘贴:使用v进入可是模式,按住hjkl进行选择,使用y复制,使用p粘贴