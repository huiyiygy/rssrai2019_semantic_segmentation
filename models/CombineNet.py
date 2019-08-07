# -*- coding:utf-8 -*-
"""
@function: 将UNet上单张图片多尺度预测的结果concat，连接2层3*3的卷积，得到融合后的预测结果
@author:HuiYi or 会意
@file: CombineNet.py
@time: 2019/8/6 下午3:09
"""
import torch.nn as nn
from models.utils.utils import init_weights


class CombineNet(nn.Module):
    def __init__(self, in_channels=96, n_classes=16,):
        super(CombineNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n_classes * 3, 3, 1, 1)
        self.conv2 = nn.Conv2d(n_classes * 3, n_classes, 3, 1, 1)

        # initialise the blocks
        for m in self.modules():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        return conv2

    def get_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                for p in m.parameters():
                    if p.requires_grad:
                        yield p
