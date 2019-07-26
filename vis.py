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
            pred_img = Image.fromarray(pred, mode='1')
            rgb_img = Image.fromarray(rgb, mode='RGB')
            filename = os.path.basename(img_path[0])
            pred_img.save(os.path.join(self.args.vis_logdir, 'raw_train_id', filename))
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
