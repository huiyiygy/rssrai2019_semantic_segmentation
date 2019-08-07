# -*- coding:utf-8 -*-
"""
@function:
@author:HuiYi or 会意
@file: vis.py.py
@time: 2019/6/23 下午7:09
"""
import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image

from dataloaders import make_data_loader
from models.backbone.UNet import UNet

from dataloaders.utils import decode_segmap


class Visualization(object):
    def __init__(self, args):
        self.args = args

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        _, _, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # Define network
        self.model = UNet(in_channels=4, n_classes=self.nclass, sync_bn=False)

        # Using cuda
        if args.cuda:
            self.model = self.model.cuda()

        if not os.path.isfile(args.checkpoint_file):
            raise RuntimeError("=> no checkpoint found at '{}'".format(args.checkpoint_file))
        checkpoint = torch.load(args.checkpoint_file)

        self.model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(args.checkpoint_file))

    def visualization(self):
        self.model.eval()
        tbar = tqdm(self.test_loader, desc='\r')
        for i, sample in enumerate(tbar):
            image = sample['image']
            img_path = sample['img_path']
            if self.args.cuda:
                image = image.cuda()
            with torch.no_grad():
                output = self.model(image)
            tbar.set_description('Vis image:')
            pred = output.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)[0]

            rgb = decode_segmap(pred, self.args.dataset)
            pred_img = Image.fromarray(pred, mode='L')
            rgb_img = Image.fromarray(rgb, mode='RGB')
            filename = os.path.basename(img_path[0])
            pred_img.save(os.path.join(self.args.vis_logdir, 'raw_train_id', filename))
            rgb_img.save(os.path.join(self.args.vis_logdir, 'vis_color', filename))

    def transform_test(self, img):
        # Normalize
        mean = (0.544650, 0.352033, 0.384602, 0.352311)
        std = (0.249456, 0.241652, 0.228824, 0.227583)
        img = np.array(img).astype(np.float32)
        img /= 255.0
        img -= mean
        img /= std
        # ToTensor
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        img = np.array([img, ])
        img = torch.from_numpy(img).float()
        return img

    def predict(self, img: np.array) -> np.array:
        img = self.transform_test(img)
        if self.args.cuda:
            img = img.cuda()
        with torch.no_grad():
            output = self.model(img)
        pred = output.data.cpu().numpy()
        return pred

    def multi_scale_predict(self):
        import cv2
        test_dir = r'/home/lab/ygy/rssrai2019/datasets/image/test_crop'
        files = os.listdir(test_dir)
        self.model.eval()
        tbar = tqdm(files, desc='\r')

        for i, filename in enumerate(tbar):
            image_predict_prob_list = []
            image_ori = Image.open(os.path.join(test_dir, filename))

            # 预测原图
            sample_ori = image_ori.copy()
            pred = self.predict(sample_ori)[0]
            ori_pred = np.argmax(pred, axis=0)
            image_predict_prob_list.append(pred)

            # 预测旋转三个
            angle_list = [90, 180, 270]
            for angle in angle_list:
                img_rotate = image_ori.rotate(angle, Image.BILINEAR)
                pred = self.predict(img_rotate)[0]
                pred = pred.transpose((1, 2, 0))
                m_rotate = cv2.getRotationMatrix2D((200, 200), 360.0-angle, 1)
                pred = cv2.warpAffine(pred, m_rotate, (400, 400))
                pred = pred.transpose((2, 0, 1))
                image_predict_prob_list.append(pred)

            # 预测竖直翻转
            img_flip = image_ori.transpose(Image.FLIP_TOP_BOTTOM)
            pred = self.predict(img_flip)[0]
            pred = cv2.flip(pred, 0)
            image_predict_prob_list.append(pred)

            # 预测水平翻转
            img_flip = image_ori.transpose(Image.FLIP_LEFT_RIGHT)
            pred = self.predict(img_flip)[0]
            pred = cv2.flip(pred, 1)
            image_predict_prob_list.append(pred)

            # 求和平均
            final_predict_prob = sum(image_predict_prob_list) / len(image_predict_prob_list)
            final_pred = np.argmax(final_predict_prob, axis=0)

            rgb_ori = decode_segmap(ori_pred, self.args.dataset)
            rgb = decode_segmap(final_pred, self.args.dataset)
            pred_img = Image.fromarray(final_pred, mode='1')
            rgb_ori_img = Image.fromarray(rgb_ori, mode='RGB')
            rgb_img = Image.fromarray(rgb, mode='RGB')
            pred_img.save(os.path.join(self.args.vis_logdir, 'raw_train_id', filename))
            rgb_ori_img.save(os.path.join(self.args.vis_logdir, 'vis_color_ori', filename))
            rgb_img.save(os.path.join(self.args.vis_logdir, 'vis_color', filename))


def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='unet',
                        choices=['unet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--dataset', type=str, default='rssrai2019',
                        choices=['rssrai2019'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--batch-size', type=int, default=1,
                        metavar='N', help='input batch size for testing (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: auto)')

    parser.add_argument('--crop-size', type=int, default=400,
                        help='crop image size')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA')

    parser.add_argument('--checkpoint_file', type=str, default=None,
                        help='put the path to checkpoint file')
    parser.add_argument('--vis_logdir', type=str, default=None,
                        help='store the vis image result dir')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    visual = Visualization(args)
    visual.visualization()


if __name__ == "__main__":
    main()
