# -*- coding:utf-8 -*-
"""
将裁剪的图片拼接回大图
"""
import os
from PIL import Image
import numpy as np

test_img_num = 10

rows, cols = 6800, 7200
# stride = 400
stride = 200
crop_rows, crop_cols = 400, 400
rows_num = (rows - crop_rows) // stride + 1
cols_num = (cols - crop_cols) // stride + 1

test_img_name = [
    'GF2_PMS1__20150902_L1A0001015646-MSS1',
    'GF2_PMS1__20150902_L1A0001015648-MSS1',
    'GF2_PMS1__20150912_L1A0001037899-MSS1',
    'GF2_PMS1__20150926_L1A0001064469-MSS1',
    'GF2_PMS1__20160327_L1A0001491484-MSS1',
    'GF2_PMS1__20160430_L1A0001553848-MSS1',
    'GF2_PMS1__20160623_L1A0001660727-MSS1',
    'GF2_PMS1__20160627_L1A0001668483-MSS1',
    'GF2_PMS1__20160704_L1A0001680853-MSS1',
    'GF2_PMS1__20160801_L1A0001734328-MSS1'
]


def stitch_test_img(color_dir, stitch_dir):
    files = os.listdir(color_dir)
    crop_num = len(files) // len(test_img_name)

    for i in range(test_img_num):
        # 拼接单张大图
        test_img_np = np.zeros((rows, cols, 3), dtype=np.uint8)
        row, col = 0, 0
        for j in range(crop_num):
            # 读取每张小图
            crop_img_name = os.path.join(color_dir, test_img_name[i]+'_'+str(j)+'.tif')
            crop_img_pil = Image.open(crop_img_name)
            crop_img_pil = crop_img_pil.resize((crop_rows, crop_cols), Image.NEAREST)
            crop_img_np = np.array(crop_img_pil, dtype=np.uint8)
            # 将小图放入大图中
            a0 = row * stride
            a1 = a0 + crop_rows
            b0 = col * stride
            b1 = b0 + crop_cols
            test_img_np[a0:a1, b0:b1, :] = crop_img_np
            # 更新行列
            col += 1
            if j != 0 and (j+1) % cols_num == 0:
                row += 1
                col = 0
        # 保存图片
        save_file_name = os.path.join(stitch_dir, test_img_name[i]+'_label.tif')
        test_img_pil = Image.fromarray(test_img_np, mode='RGB')
        test_img_pil.save(save_file_name)


if __name__ == "__main__":
    vis_color_dir = '/home/lab/ygy/rssrai2019/rssrai2019_semantic_segmentation/run/rssrai2019/combine_net/vis_log/vis_color'
    stitch_img_dir = '/home/lab/ygy/rssrai2019/rssrai2019_semantic_segmentation/run/rssrai2019/combine_net/vis_log/stitch_img'
    stitch_test_img(vis_color_dir, stitch_img_dir)

