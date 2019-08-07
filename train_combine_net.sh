#!/usr/bin/env bash

python train_combine_net.py --learn-rate 0.001 --weight-decay 0 --epochs 100 --base-size 400 --crop-size 400 --gpu-ids 0 --checkname combine_net --eval-interval 1 --dataset rssrai2019 --checkpoint_file /home/lab/ygy/rssrai2019/rssrai2019_semantic_segmentation/run/rssrai2019/unet/experiment_2/checkpoint.pth.tar