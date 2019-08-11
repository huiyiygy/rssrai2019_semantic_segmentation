# -*- coding:utf-8 -*-
import numpy as np
import os
from PIL import Image


def crop_tiff_image(filename, source_dir, stride, crop_size, target_dir, postfix):
    img_pil = Image.open(os.path.join(source_dir, filename))
    img_np = np.array(img_pil, dtype=np.uint8)

    rows, cols = img_np.shape[0], img_np.shape[1]
    crop_rows, crop_cols = crop_size[0], crop_size[1]

    if (rows - crop_rows) % stride != 0 or (cols - crop_cols) % stride != 0:
        raise ValueError('Inappropriate crop size {}' % crop_size)

    rows_num = (rows - crop_rows) // stride + 1
    cols_num = (cols - crop_cols) // stride + 1

    n = 0
    for i in range(rows_num):
        for j in range(cols_num):
            crop_img_pil = None
            if len(img_np.shape) == 2:
                crop_img_np = img_np[i * stride:i * stride + crop_rows, j * stride:j * stride + crop_cols]
                crop_img_pil = Image.fromarray(crop_img_np)
            else:
                crop_img_np = img_np[i * stride:i * stride + crop_rows, j * stride:j * stride + crop_cols, :]
                if img_np.shape[2] == 3:
                    crop_img_pil = Image.fromarray(crop_img_np, mode='RGB')
                elif img_np.shape[2] == 4:
                    crop_img_pil = Image.fromarray(crop_img_np, mode='CMYK')
            crop_img_filename = os.path.splitext(filename)[0] + '_' + str(n) + postfix
            # 将原始训练集、验证集分割图片打散, 组成新的训练集、验证集
            # if n % 5 == 4:
            #     crop_img_pil.save(os.path.join(val_mix_dir, crop_img_filename))
            # else:
            #     crop_img_pil.save(os.path.join(train_mix_dir, crop_img_filename))
            crop_img_pil.save(os.path.join(target_dir, crop_img_filename))
            n += 1


if __name__ == "__main__":
    # source_dir = r'/home/lab/ygy/rssrai2019/datasets/label/train_id_image'
    # target_dir = r'/home/lab/ygy/rssrai2019/datasets/label/train_id_image_crop'
    # source_dir = r'/home/lab/ygy/rssrai2019/datasets/label/val_id_image'
    # target_dir = r'/home/lab/ygy/rssrai2019/datasets/label/val_id_image_crop'
    # stride = 400
    # postfix = '.png'

    # source_dir = r'/home/lab/ygy/rssrai2019/datasets/image/train'
    # target_dir = r'/home/lab/ygy/rssrai2019/datasets/image/train_crop'
    # source_dir = r'/home/lab/ygy/rssrai2019/datasets/image/val'
    # target_dir = r'/home/lab/ygy/rssrai2019/datasets/image/val_crop'
    # train_mix_dir = r'/home/lab/ygy/rssrai2019/datasets/image/train_mix'
    # val_mix_dir = r'/home/lab/ygy/rssrai2019/datasets/image/val_mix'
    # source_dir = r'/home/lab/ygy/rssrai2019/datasets/image/test'
    # target_dir = r'/home/lab/ygy/rssrai2019/datasets/image/test_crop'
    # source_dir = r'/home/lab/ygy/rssrai2019/datasets/label/train'
    # target_dir = r'/home/lab/ygy/rssrai2019/datasets/label/train_crop'
    # source_dir = r'/home/lab/ygy/rssrai2019/datasets/label/val'
    # target_dir = r'/home/lab/ygy/rssrai2019/datasets/label/val_crop'
    # train_mix_dir = r'/home/lab/ygy/rssrai2019/datasets/label/train_mix'
    # val_mix_dir = r'/home/lab/ygy/rssrai2019/datasets/label/val_mix'
    # stride = 400
    source_dir = r'/home/lab/ygy/rssrai2019/datasets/image/test'
    target_dir = r'/home/lab/ygy/rssrai2019/datasets/image/test_overlay_crop'
    stride = 200

    postfix = '.tif'

    crop_size = [400, 400]

    n = 1
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if os.path.splitext(file)[1] == postfix:
                print('cropping the %d th image, filename=%s' % (n, file))
                crop_tiff_image(file, root, stride, crop_size, target_dir, postfix)
                n += 1
    print('crop finished')
