# -*- coding:utf-8 -*-
"""
@function:
@author:HuiYi or 会意
@file: train_combine_net.py
@time: 2019/8/6 下午3:20
"""
import argparse
import os
import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm

from utils.saver import Saver
from utils.summaries import TensorboardSummary
from models.backbone.UNet import UNet
from models.backbone.UNetNested import UNetNested
from models.CombineNet import CombineNet
from utils.loss import SegmentationLosses
from utils.metrics import Evaluator


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        self.nclass = 16
        # Define network
        self.unet_model = UNet(in_channels=4, n_classes=self.nclass)
        self.unetNested_model = UNetNested(in_channels=4, n_classes=self.nclass)
        self.combine_net_model = CombineNet(in_channels=192, n_classes=self.nclass)

        train_params = [{'params': self.combine_net_model.get_params()}]
        # Define Optimizer
        self.optimizer = torch.optim.Adam(train_params, self.args.learn_rate, weight_decay=args.weight_decay, amsgrad=True)

        self.criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode=args.loss_type)

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)

        # Using cuda
        if args.cuda:
            self.unet_model = self.unet_model.cuda()
            self.unetNested_model = self.unetNested_model.cuda()
            self.combine_net_model = self.combine_net_model.cuda()

        # Load Unet checkpoint
        if not os.path.isfile(args.unet_checkpoint_file):
            raise RuntimeError("=> no Unet checkpoint found at '{}'".format(args.unet_checkpoint_file))
        checkpoint = torch.load(args.unet_checkpoint_file)
        self.unet_model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded Unet checkpoint '{}'".format(args.unet_checkpoint_file))

        # Load UNetNested checkpoint
        if not os.path.isfile(args.unetNested_checkpoint_file):
            raise RuntimeError("=> no UNetNested checkpoint found at '{}'".format(args.unetNested_checkpoint_file))
        checkpoint = torch.load(args.unetNested_checkpoint_file)
        self.unetNested_model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded UNetNested checkpoint '{}'".format(args.unetNested_checkpoint_file))

        # Resuming combineNet checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no combineNet checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.combine_net_model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.combine_net_model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded combineNet checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        print('[Epoch: %d, previous best = %.4f]' % (epoch, self.best_pred))
        train_loss = 0.0
        self.combine_net_model.train()
        self.evaluator.reset()
        num_img_tr = len(train_files)
        tbar = tqdm(train_files, desc='\r')

        for i, filename in enumerate(tbar):
            image = Image.open(os.path.join(train_dir, filename))
            label = Image.open(os.path.join(train_label_dir, os.path.basename(filename)[:-4] + '_labelTrainIds.png'))
            label = np.array(label).astype(np.float32)
            label = label.reshape((1, 400, 400))
            label = torch.from_numpy(label).float()
            label = label.cuda()

            # UNet_multi_scale_predict
            unt_pred = self.unet_multi_scale_predict(image)

            # UNetNested_multi_scale_predict
            unetnested_pred = self.unetnested_multi_scale_predict(image)

            net_input = torch.cat([unt_pred, unetnested_pred], 1)

            self.optimizer.zero_grad()
            output = self.combine_net_model(net_input)
            loss = self.criterion(output, label)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.5f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

        pred = output.data.cpu().numpy()
        label = label.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        # Add batch sample into evaluator
        self.evaluator.add_batch(label, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('train/mIoU', mIoU, epoch)
        self.writer.add_scalar('train/Acc', Acc, epoch)
        self.writer.add_scalar('train/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('train/fwIoU', FWIoU, epoch)
        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)

        print('train validation:')
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % train_loss)
        print('---------------------------------')

    def validation(self, epoch):
        test_loss = 0.0
        self.combine_net_model.eval()
        self.evaluator.reset()
        tbar = tqdm(val_files, desc='\r')
        num_img_val = len(val_files)

        for i, filename in enumerate(tbar):
            image = Image.open(os.path.join(val_dir, filename))
            label = Image.open(os.path.join(val_label_dir, os.path.basename(filename)[:-4] + '_labelTrainIds.png'))
            label = np.array(label).astype(np.float32)
            label = label.reshape((1, 400, 400))
            label = torch.from_numpy(label).float()
            label = label.cuda()

            # UNet_multi_scale_predict
            unt_pred = self.unet_multi_scale_predict(image)

            # UNetNested_multi_scale_predict
            unetnested_pred = self.unetnested_multi_scale_predict(image)

            net_input = torch.cat([unt_pred, unetnested_pred], 1)

            with torch.no_grad():
                output = self.combine_net_model(net_input)
            loss = self.criterion(output, label)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.5f' % (test_loss / (i + 1)))
            self.writer.add_scalar('val/total_loss_iter', loss.item(), i + num_img_val * epoch)
            pred = output.data.cpu().numpy()
            label = label.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(label, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('test validation:')
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)
        print('====================================')

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.combine_net_model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

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

    def unetnested_predict(self, img: Image) -> torch.Tensor:
        img = self.transform_test(img)
        if self.args.cuda:
            img = img.cuda()
        with torch.no_grad():
            output = self.unetNested_model(img)
        return output

    def unetnested_multi_scale_predict(self, image_ori: Image):
        self.unetNested_model.eval()

        # 预测原图
        sample_ori = image_ori.copy()
        output_ori = self.unetnested_predict(sample_ori)

        # 预测旋转三个
        angle_list = [90, 180, 270]
        for angle in angle_list:
            img_rotate = image_ori.rotate(angle, Image.BILINEAR)
            output = self.unetnested_predict(img_rotate)
            pred = output.data.cpu().numpy()[0]
            pred = pred.transpose((1, 2, 0))
            m_rotate = cv2.getRotationMatrix2D((200, 200), 360.0 - angle, 1)
            pred = cv2.warpAffine(pred, m_rotate, (400, 400))
            pred = pred.transpose((2, 0, 1))
            output = torch.from_numpy(np.array([pred, ])).float()
            output_ori = torch.cat([output_ori, output.cuda()], 1)

        # 预测竖直翻转
        img_flip = image_ori.transpose(Image.FLIP_TOP_BOTTOM)
        output = self.unetnested_predict(img_flip)
        pred = output.data.cpu().numpy()[0]
        pred = pred.transpose((1, 2, 0))
        pred = cv2.flip(pred, 0)
        pred = pred.transpose((2, 0, 1))
        output = torch.from_numpy(np.array([pred, ])).float()
        output_ori = torch.cat([output_ori, output.cuda()], 1)

        # 预测水平翻转
        img_flip = image_ori.transpose(Image.FLIP_LEFT_RIGHT)
        output = self.unetnested_predict(img_flip)
        pred = output.data.cpu().numpy()[0]
        pred = pred.transpose((1, 2, 0))
        pred = cv2.flip(pred, 1)
        pred = pred.transpose((2, 0, 1))
        output = torch.from_numpy(np.array([pred, ])).float()
        output_ori = torch.cat([output_ori, output.cuda()], 1)

        return output_ori

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
                        help='dataset name (default: rssrai2019)')
    parser.add_argument('--workers', type=int, default=2,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=400,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=400,
                        help='crop image size')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N',
                        help='start epochs (default:0)')

    # optimizer params
    parser.add_argument('--learn-rate', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--unet_checkpoint_file', type=str, default=None,
                        help='put the path to Unet checkpoint file')
    parser.add_argument('--unetNested_checkpoint_file', type=str, default=None,
                        help='put the path to UNetNested checkpoint file')
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to combineNet resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {'rssrai2019': 100}
        args.epochs = epoches[args.dataset.lower()]

    if args.learn_rate is None:
        lrs = {'rssrai2019': 0.001}
        args.learn_rate = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    print('====================================')
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()


if __name__ == "__main__":
    train_dir = r'/home/lab/ygy/rssrai2019/datasets/image/train_mix'
    train_label_dir = r'/home/lab/ygy/rssrai2019/datasets/label/train_mix_id_image'
    val_dir = r'/home/lab/ygy/rssrai2019/datasets/image/val_mix'
    val_label_dir = r'/home/lab/ygy/rssrai2019/datasets/label/val_mix_id_image'

    train_files = os.listdir(train_dir)
    val_files = os.listdir(val_dir)
    main()
