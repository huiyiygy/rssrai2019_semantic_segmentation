# -*- coding:utf-8 -*-
import os
from PIL import Image
from torch.utils import data
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr


class RssraiSegmentation(data.Dataset):
    NUM_CLASSES = 16

    def __init__(self, args, root=Path.db_root_dir('rssrai2019'), split="train"):
        self.root = root
        self.split = split
        self.args = args
        self.files = {}

        self.images_base = os.path.join(self.root, 'image', self.split+'_mix')
        self.annotations_base = os.path.join(self.root, 'label', self.split+'_mix_id_image')

        self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix='.tif')

        self.classes = [0, 1, 2, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        self.class_names = ['其他类别', '水   田', '水 浇地', '旱 耕地', '园   地', '乔木林地', '灌木林地', '天然草地', '人工草地',
                            '工业用地', '城市住宅', '村镇住宅', '交通运输', '河   流', '湖   泊', '坑   塘']

        self.ignore_index = 255

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_path = self.files[self.split][index].rstrip()
        _img = Image.open(img_path)

        if self.split != 'test':
            lbl_path = os.path.join(self.annotations_base, os.path.basename(img_path)[:-4] + '_labelTrainIds.png')
            _target = Image.open(lbl_path)
            sample = {'image': _img, 'label': _target}
        else:
            sample = {'image': _img, 'label': _img, 'img_path': img_path}  # We do not have test label

        if self.split == 'train':
            return self.transform_train(sample)
        elif self.split == 'val':
            return self.transform_val(sample)
        elif self.split == 'test':
            return self.transform_test(sample)

    @staticmethod
    def recursive_glob(rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]

    def transform_train(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomVerticalFlip(),
            # tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=255),
            # tr.FixedResize(size=self.args.crop_size),
            tr.RandomRotate(),
            tr.RandomGammaTransform(),
            tr.RandomGaussianBlur(),
            tr.RandomNoise(),
            tr.Normalize(mean=(0.544650, 0.352033, 0.384602, 0.352311), std=(0.249456, 0.241652, 0.228824, 0.227583)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            # tr.FixScaleCrop(crop_size=self.args.crop_size),
            # tr.FixedResize(size=self.args.crop_size),
            tr.Normalize(mean=(0.544650, 0.352033, 0.384602, 0.352311), std=(0.249456, 0.241652, 0.228824, 0.227583)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_test(self, sample):
        composed_transforms = transforms.Compose([
            # tr.FixedResize(size=self.args.crop_size),
            tr.Normalize(mean=(0.544650, 0.352033, 0.384602, 0.352311), std=(0.249456, 0.241652, 0.228824, 0.227583)),
            tr.ToTensor()])

        return composed_transforms(sample)
