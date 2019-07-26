# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import torch


def decode_seg_map_sequence(label_masks, dataset='rssrai2019'):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks


def decode_segmap(label_mask, dataset, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        dataset
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == 'rssrai2019':
        n_classes = 16
        label_colours = get_rssrai_labels()
    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3), dtype=np.uint8)
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb


def get_rssrai_labels():
    return np.array([
        [0, 0, 0],
        [0, 200, 0],
        [150, 250, 0],
        [150, 200, 150],
        [200, 0, 200],
        [150, 0, 250],
        [150, 150, 250],
        [250, 200, 0],
        [200, 200, 0],
        [200, 0, 0],
        [250, 0, 150],
        [200, 150, 150],
        [250, 150, 150],
        [0, 0, 200],
        [0, 150, 200],
        [0, 200, 250]])
