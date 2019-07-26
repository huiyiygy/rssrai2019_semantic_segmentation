# -*- coding:utf-8 -*-
import os
from PIL import Image
import numpy as np

filepath = '/home/lab/ygy/rssrai2019/datasets/image/temp'  # 数据集目录
pathDir = os.listdir(filepath)

N_channel = 0
R_channel = 0
G_channel = 0
B_channel = 0
for idx in range(len(pathDir)):
    filename = pathDir[idx]
    img = Image.open(os.path.join(filepath, filename))
    img_np = np.array(img, dtype=np.uint8) / 255
    N_channel = N_channel + np.sum(img_np[:, :, 0])
    R_channel = R_channel + np.sum(img_np[:, :, 1])
    G_channel = G_channel + np.sum(img_np[:, :, 2])
    B_channel = B_channel + np.sum(img_np[:, :, 3])

num = len(pathDir) * 400 * 400  # 这里（400,400）是每幅图片的大小，所有图片尺寸都一样
N_mean = N_channel / num
R_mean = R_channel / num
G_mean = G_channel / num
B_mean = B_channel / num

N_channel = 0
R_channel = 0
G_channel = 0
B_channel = 0
for idx in range(len(pathDir)):
    filename = pathDir[idx]
    img = Image.open(os.path.join(filepath, filename))
    img_np = np.array(img, dtype=np.uint8) / 255
    N_channel = N_channel + np.sum((img_np[:, :, 0] - N_mean) ** 2)
    R_channel = R_channel + np.sum((img_np[:, :, 1] - R_mean) ** 2)
    G_channel = G_channel + np.sum((img_np[:, :, 2] - G_mean) ** 2)
    B_channel = B_channel + np.sum((img_np[:, :, 3] - B_mean) ** 2)

N_var = N_channel / num
R_var = R_channel / num
G_var = G_channel / num
B_var = B_channel / num

N_std = np.sqrt(N_var)
R_std = np.sqrt(R_var)
G_std = np.sqrt(G_var)
B_std = np.sqrt(B_var)

# mean = (0.544650, 0.352033, 0.384602, 0.352311)
print("N_mean is %f, R_mean is %f, G_mean is %f, B_mean is %f" % (N_mean, R_mean, G_mean, B_mean))
# var = (0.062228, 0.058396, 0.052360, 0.051794)
print("N_var is %f, R_var is %f, G_var is %f, B_var is %f" % (N_var, R_var, G_var, B_var))
# std = (0.249456, 0.241652, 0.228824, 0.227583)
print("N_std is %f, R_std is %f, G_std is %f, B_std is %f" % (N_std, R_std, G_std, B_std))
