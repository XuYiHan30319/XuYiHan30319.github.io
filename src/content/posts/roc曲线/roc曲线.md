---
title: roc曲线
published: 2024-07-16
description: ''
image: ''
tags: [模型优化]
category: '机器学习'
draft: false 
---

![image-20240716143436985](https://p.ipic.vip/7cmuxm.png)

ROC曲线实际上是多个混淆矩阵的结果组合，以疾病检测为例，这是一个有监督的二分类模型，模型对每个样本的预测结果为一个概率值，我们需要从中选取一个阈值来判断健康与否。定义好一个阈值之后，超过此阈值定义为不健康，低于此阈值定义为健康，就可以得出混淆矩阵。而如果在上述模型中没有定义好阈值，而是将模型预测结果从高到低排序(排不排序都一样，因为我们需要用作作图的TPR和FPR都是根据这些概率值计算出来的，现在不排序，等据图画图的时候也得排序)，将每次概率值依次作为阈值，那么就可以得到多个混淆矩阵。对于每个混淆矩阵，我们计算两个指标TPR和FPR，以FPR为轴，TPR为y轴画图，就得到了ROC曲线。

```python
# roc曲线阈值确定,用于确定区分点
def multi_label_roc(labels, predictions, num_classes, pos_label=1):
    fprs = []
    tprs = []
    thresholds = []
    thresholds_optimal = []
    aucs = []
    if len(predictions.shape) == 1:
        predictions = predictions[:, None]
    if labels.ndim == 1:
        labels = np.expand_dims(labels, axis=-1)
    for c in range(0, num_classes):
        label = labels[:, c]
        prediction = predictions[:, c]
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(
            fpr, tpr, threshold
        )
        # c_auc = roc_auc_score(label, prediction)
        try:
            c_auc = roc_auc_score(label, prediction)
            print("ROC AUC score:", c_auc)
        except ValueError as e:
            if (
                str(e)
                == "Only one class present in y_true. ROC AUC score is not defined in that case."
            ):
                print(
                    "ROC AUC score is not defined when only one class is present in y_true. c_auc is set to 1."
                )
                c_auc = 1
            else:
                raise e

        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
    return aucs, thresholds, thresholds_optimal


def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

```

比如在二分类任务中,默认的区分点为0.5,为了寻找最优的分割阈值,我们可以使用ROC曲线来计算得到最优的分割点.我们需要找到最优的平衡点
