# -*- coding:utf-8 -*-
# Converts the color images of the rssrai2019 dataset
# to labelTrainIds images, where pixel values encode ground truth classes.
#
# The rssrai2019 downloads already include such images
#   a) *color.tif             : the class is encoded by its color
#
# With this tool, you can generate option
#   b) *labelTrainIds.png     : the class is encoded by its training ID
# This encoding might come handy for training purposes. You can use
# the file labes.py to define the training IDs that suit your needs.
# Note however, that once you submit or evaluate results, the regular
# IDs are needed.
#
# Uses the mapping defined in 'labels.py'
#
import numpy as np
import os
from dataloaders.datasets.utils.labels import color2label

from PIL import Image


def create_train_id_imgs(filename, source_dir, target_dir):
    img_pil = Image.open(os.path.join(source_dir, filename))
    img_np = np.array(img_pil, dtype=np.uint8)

    rows, cols = img_np.shape[0], img_np.shape[1]
    train_id_img_np = np.zeros((rows, cols), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            color = (img_np[i, j, 0], img_np[i, j, 1], img_np[i, j, 2])
            train_id = color2label[color].trainId
            train_id_img_np[i, j] = train_id
    train_id_img = Image.fromarray(train_id_img_np)
    train_id_img_filename = filename[:-4] + '_labelTrainIds.png'
    train_id_img.save(os.path.join(target_dir, train_id_img_filename))


if __name__ == "__main__":
    # source_dir = r'/home/lab/ygy/rssrai2019/datasets/label/train_crop'
    # target_dir = r'/home/lab/ygy/rssrai2019/datasets/label/train_crop_id_image'
    # source_dir = r'/home/lab/ygy/rssrai2019/datasets/label/val_crop'
    # target_dir = r'/home/lab/ygy/rssrai2019/datasets/label/val_crop_id_image'

    # source_dir = r'/home/lab/ygy/rssrai2019/datasets/label/train_mix'
    # target_dir = r'/home/lab/ygy/rssrai2019/datasets/label/train_mix_id_image'
    source_dir = r'/home/lab/ygy/rssrai2019/datasets/label/val_mix'
    target_dir = r'/home/lab/ygy/rssrai2019/datasets/label/val_mix_id_image'

    n = 1
    for root, _, files in os.walk(source_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.tif':
                print('creating the %d th labelTrainIds image, filename=%s' % (n, file))
                create_train_id_imgs(file, source_dir, target_dir)
                n += 1
    print('create finished')
