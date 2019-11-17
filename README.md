# 2019年遥感图像稀疏表征与智能分析竞赛-语义分割组
比赛官网：[http://rscup.bjxintong.com.cn/#/theme/3/](http://rscup.bjxintong.com.cn/#/theme/3)

代码主要基于我的另外一个仓库：https://github.com/huiyiygy/pytorch-deeplab-xception/tree/huiyiygy 修改而来。

最终提交kappa系数最高得分为0.37187

# 实验记录

| 序号 | 实验序号      | 实验更改内容                                                 | 结果                       |
| ---- | ------------- | ------------------------------------------------------------ | -------------------------- |
| 1    | Deeplab 0     | SGD --lr 0.01 --out-stride 16 --epochs 200 --batch-size 8  --out-stride 16 | 0.25705                    |
| 2    | Deeplab 1 / 2 | 改变数据集分布，将原始train/val打散，重新组合成新的train/val  batch-size 10  --out-stride 16 | 0.25024                    |
| 3    | Deeplab 3 / 4 | --out-stride 8 --epochs 100 --batch-size 4                   | 0.12774                    |
| 4    | Deeplab 5     | 在训练阶段取train数据时，重新添加随机尺度裁剪步骤  混合数据集  --out-stride 16 --epochs 200 --batch-size 10 | 0.19396                    |
| 5    | Deeplab 6     | 使用自带的预训练模型进行训练  混合数据集                     | 0.20624                    |
| 6    | Deeplab 7     | 使用随机尺度裁剪对原始数据集进行增强，扩展图片至1W张, 使用重新生成的数据集进行训练  --weight-decay 0.001 --out-stride 16 --epochs 100 --batch-size 14 | 精简后的模型依旧严重过拟合 |
| 7    | UNet 0        | SGD   --lr 0.01 --weight-decay 0.001 --epochs 200 --batch-size 32 | 0.19                       |
| 8    | UNet 1        | using Adam no weight-decay  --learn-rate 0.001 --weight-decay 0 --epochs 1000 --batch-size 32 | 0.28297                    |
| 9    | UNet 2        | 将编码器模块中每块增加一层卷积层，并在下采样层最后添加dropout=0.5, RandomGammaTransform, RandomBilateralFilter, RandomNoise   --batch-size 20 | 0.36428                    |
| 10   | UNet 3        | 在实验2的基础上add weight-decay amsgrad,  当epoch为总数的50%,80%时，将学习率缩小10倍   --weight-decay 1e-4 | 0.3582                     |
| 11   | UNet 4        | 实验3 L2系数过大导致精度过低，将其继续缩小10倍。学习率更新改为每次0.3倍，增加epoch数量 | 0.29755                    |
| 12   | UNet++        | 使用UNet++网络训练  --learn-rate 0.001 --weight-decay 0 --epochs 1000 --batch-size 12 | 0.35381                    |
| 13   | CombineNet 1  | 单Unet模型，四个角度、上下水平翻转，6张图预测                | 0.37187                    |
| 14   | CombineNet 2  | Unet与Unet++ ，多角度预测                                    | 没来得及提交                       |

# 总结

1. （请先容我抱怨几句）官方提供的数据标注质量太差了，数据存在严重不均衡、错标、漏标现象。同时提供的数据量也较少（一共10张图片，我能训练出花来？）。
2. 使用Deeplab训练，普遍存在严重的过拟合现象。原因：模型复杂度太高，样本数据太少。就算加入正则化项和数据增强，效果也没有提升。
3. 原生UNet网络可以很好的避免过拟合问题，可以以此为baseline，逐步添加更多的层以及其他训练技巧，提高精度。

# 后续可以尝试的方法
* LovaszSoftmax Loss
* Test阶段小图预测后拼接回大图时的拼接方法可以进一步优化，目前我用的是400大小，200步长去裁剪，使得每张图片有一半是重合的，这样做一定程度上减轻了边界黑边的问题，不过还是存在。
* 有遥感图像专业知识背景的同学，可以尝试用envi对原始标注图片进行修改，我和某位大佬沟通过，他只通过改原始标注图片，然后用相同的方法训练，分数就提高了10个点。
