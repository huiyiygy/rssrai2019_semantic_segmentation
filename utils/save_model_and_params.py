# -*- coding:utf-8 -*-
"""
@function: 读取单独保存的模型参数，将其与模型结构一起重新保存
@author:HuiYi or 会意
@file: vis.py.py
@time: 2019/7/30 下午7:00
"""
import torch
from models.backbone.UNet import UNet

model_path_list = [
    '/home/lab/ygy/rssrai2019/rssrai2019_semantic_segmentation/run/rssrai2019/unet/experiment_0/checkpoint.pth.tar',
    '/home/lab/ygy/rssrai2019/rssrai2019_semantic_segmentation/run/rssrai2019/unet/experiment_1/checkpoint.pth.tar',
    '/home/lab/ygy/rssrai2019/rssrai2019_semantic_segmentation/run/rssrai2019/unet/experiment_2/checkpoint.pth.tar'
]

if __name__ == '__main__':
    model = UNet(in_channels=4, n_classes=16, sync_bn=False)
    model = model.cuda()
    param = '/home/lab/ygy/rssrai2019/rssrai2019_semantic_segmentation/run/rssrai2019/unet/experiment_0/checkpoint.pth.tar'
    checkpoint = torch.load(param)
    model.load_state_dict(checkpoint['state_dict'])
    torch.save(model, '/home/lab/ygy/rssrai2019/rssrai2019_semantic_segmentation/run/rssrai2019/unet/experiment_0/model_and_param.pth.tar')
    print('save finish')

    # load
    # model = torch.load('/home/lab/ygy/rssrai2019/rssrai2019_semantic_segmentation/run/rssrai2019/unet/experiment_1/model_and_param.pth.tar')
    # params = model.state_dict()
    # print('load')
