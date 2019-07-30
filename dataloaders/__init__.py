# -*- coding:utf-8 -*-
from dataloaders.datasets import rssrai
from torch.utils.data import DataLoader


def make_data_loader(args, **kwargs):
    if args.dataset == 'rssrai2019':
        train_set = rssrai.RssraiSegmentation(args, split='train')
        val_set = rssrai.RssraiSegmentation(args, split='val')
        test_set = rssrai.RssraiSegmentation(args, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError
