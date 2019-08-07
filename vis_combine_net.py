# -*- coding:utf-8 -*-
"""
@function:
@author:HuiYi or 会意
@file: vis_combine_net.py
@time: 2019/8/7 上午11:25
"""
import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
import cv2
from PIL import Image

from models.backbone.UNet import UNet
from models.CombineNet import CombineNet

from dataloaders.utils import decode_segmap


class Visualization:
    def __init__(self, args):
        self.args = args

        self.nclass = 16
        # Define network
        self.unet_model = UNet(in_channels=4, n_classes=self.nclass)
        self.combine_net_model = CombineNet(in_channels=96, n_classes=self.nclass)

        # Using cuda
        if args.cuda:
            self.unet_model = self.unet_model.cuda()
            self.combine_net_model = self.combine_net_model.cuda()

        # Load model
        if not os.path.isfile(args.unet_checkpoint_file):
            raise RuntimeError("=> no unet checkpoint found at '{}'".format(args.unet_checkpoint_file))
        checkpoint = torch.load(args.unet_checkpoint_file)
        self.unet_model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded unet checkpoint '{}'".format(args.unet_checkpoint_file))

        if not os.path.isfile(args.combine_net_checkpoint_file):
            raise RuntimeError("=> no combine net checkpoint found at '{}'".format(args.combine_net_checkpoint_file))
        checkpoint = torch.load(args.combine_net_checkpoint_file)
        self.combine_net_model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded combine net checkpoint '{}'".format(args.combine_net_checkpoint_file))

    def visualization(self):
        self.combine_net_model.eval()
        tbar = tqdm(test_files, desc='\r')

        for i, filename in enumerate(tbar):
            image = Image.open(os.path.join(test_dir, filename))

            # UNet_multi_scale_predict
            unt_pred = self.unet_multi_scale_predict(image)

            with torch.no_grad():
                output = self.combine_net_model(unt_pred)
            pred = output.data.cpu().numpy()[0]
            pred = np.argmax(pred, axis=0)

            rgb = decode_segmap(pred, self.args.dataset)

            pred_img = Image.fromarray(pred, mode='L')
            rgb_img = Image.fromarray(rgb, mode='RGB')

            pred_img.save(os.path.join(self.args.vis_logdir, 'raw_train_id', filename))
            rgb_img.save(os.path.join(self.args.vis_logdir, 'vis_color', filename))

    def unet_multi_scale_predict(self, image_ori: Image):
        self.unet_model.eval()

        # 预测原图
        sample_ori = image_ori.copy()
        output_ori = self.unet_predict(sample_ori)

        # 预测旋转三个
        angle_list = [90, 180, 270]
        for angle in angle_list:
            img_rotate = image_ori.rotate(angle, Image.BILINEAR)
            output = self.unet_predict(img_rotate)
            pred = output.data.cpu().numpy()[0]
            pred = pred.transpose((1, 2, 0))
            m_rotate = cv2.getRotationMatrix2D((200, 200), 360.0 - angle, 1)
            pred = cv2.warpAffine(pred, m_rotate, (400, 400))
            pred = pred.transpose((2, 0, 1))
            output = torch.from_numpy(np.array([pred, ])).float()
            output_ori = torch.cat([output_ori, output.cuda()], 1)

        # 预测竖直翻转
        img_flip = image_ori.transpose(Image.FLIP_TOP_BOTTOM)
        output = self.unet_predict(img_flip)
        pred = output.data.cpu().numpy()[0]
        pred = pred.transpose((1, 2, 0))
        pred = cv2.flip(pred, 0)
        pred = pred.transpose((2, 0, 1))
        output = torch.from_numpy(np.array([pred, ])).float()
        output_ori = torch.cat([output_ori, output.cuda()], 1)

        # 预测水平翻转
        img_flip = image_ori.transpose(Image.FLIP_LEFT_RIGHT)
        output = self.unet_predict(img_flip)
        pred = output.data.cpu().numpy()[0]
        pred = pred.transpose((1, 2, 0))
        pred = cv2.flip(pred, 1)
        pred = pred.transpose((2, 0, 1))
        output = torch.from_numpy(np.array([pred, ])).float()
        output_ori = torch.cat([output_ori, output.cuda()], 1)

        return output_ori

    def unet_predict(self, img: Image) -> torch.Tensor:
        img = self.transform_test(img)
        if self.args.cuda:
            img = img.cuda()
        with torch.no_grad():
            output = self.unet_model(img)
        return output

    @staticmethod
    def transform_test(img):
        # Normalize
        mean = (0.544650, 0.352033, 0.384602, 0.352311)
        std = (0.249456, 0.241652, 0.228824, 0.227583)
        img = np.array(img).astype(np.float32)
        img /= 255.0
        img -= mean
        img /= std
        # ToTensor
        img = img.transpose((2, 0, 1))
        img = np.array([img, ])
        img = torch.from_numpy(img).float()
        return img


def main():
    parser = argparse.ArgumentParser(description="PyTorch CombineNet Training")
    parser.add_argument('--backbone', type=str, default='combine_net',
                        choices=['combine_net'],
                        help='backbone name (default: combine_net)')
    parser.add_argument('--dataset', type=str, default='rssrai2019',
                        choices=['rssrai2019'],
                        help='dataset name (default: pascal)')

    parser.add_argument('--unet_checkpoint_file', type=str, default=None,
                        help='put the path to UNet checkpoint file')
    parser.add_argument('--combine_net_checkpoint_file', type=str, default=None,
                        help='put the path to combineNet checkpoint file')
    parser.add_argument('--vis_logdir', type=str, default=None,
                        help='store the vis image result dir')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    visual = Visualization(args)
    visual.visualization()


if __name__ == "__main__":
    test_dir = r'/home/lab/ygy/rssrai2019/datasets/image/test_mix'

    test_files = os.listdir(test_dir)

    main()
