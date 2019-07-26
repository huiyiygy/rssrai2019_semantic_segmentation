# -*- coding:utf-8 -*-
"""
@function:
@author:HuiYi or 会意
@file: RandomCropTiffImage.py
@time: 2019/7/23 上午9:52
"""
import numpy as np
import os
from PIL import Image, ImageOps
from tqdm import tqdm


def random_scale_crop(filename, base_size, crop_size, crop_num, path_dict, postfix):
    img_raw = Image.open(os.path.join(path_dict['img_dir'], filename))
    color_img_raw = Image.open(os.path.join(path_dict['color_dir'], filename))
    id_img_raw = Image.open(os.path.join(path_dict['id__dir'], filename[:-3] + 'png'))

    for i in tqdm(range(crop_num)):
        img = img_raw.copy()
        color_img = color_img_raw.copy()
        id_img = id_img_raw.copy()

        # random scale (short edge)
        short_size = np.random.randint(int(base_size * 0.5), int(base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        color_img = color_img.resize((ow, oh), Image.BILINEAR)
        id_img = id_img.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            color_img = ImageOps.expand(color_img, border=(0, 0, padw, padh), fill=0)
            id_img = ImageOps.expand(id_img, border=(0, 0, padw, padh), fill=255)
        # random crop crop_size
        w, h = img.size
        x1 = np.random.randint(0, w - crop_size)
        y1 = np.random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        color_img = color_img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        id_img = id_img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # set filename
        img_filename = filename[:-4] + '_%04d.tif' % i
        id_img_filename = filename[:-4] + '_%04d_labelTrainIds.png' % i
        # save file
        img.save(os.path.join(path_dict['img_random_crop_dir'], img_filename))
        color_img.save(os.path.join(path_dict['color_random_crop_dir'], img_filename))
        id_img.save(os.path.join(path_dict['id_random_crop_dir'], id_img_filename))


if __name__ == "__main__":
    paths = {'img_dir': r'/home/lab/ygy/rssrai2019/datasets/image/train',
             'img_random_crop_dir': r'/home/lab/ygy/rssrai2019/datasets/image/train_random_crop',
             'color_dir': r'/home/lab/ygy/rssrai2019/datasets/label/train',
             'color_random_crop_dir': r'/home/lab/ygy/rssrai2019/datasets/label/train_random_crop',
             'id__dir': r'/home/lab/ygy/rssrai2019/datasets/label/train_id_image',
             'id_random_crop_dir': r'/home/lab/ygy/rssrai2019/datasets/label/train_random_crop_id_image'
             }

    postfix = '.tif'
    base_size = 6800
    crop_size = 400

    img_crop_num = 1250

    n = 1
    for root, dirs, files in os.walk(paths['img_dir']):
        for file in files:
            print('cropping the %d th image, filename=%s' % (n, file))
            random_scale_crop(file, base_size, crop_size, img_crop_num, paths, postfix)
            n += 1
    print('crop finished')
