#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1 python train.py --lr 0.01 --weight-decay 0.001 --epochs 200 --batch-size 32 --test-batch-size 32 --base-size 400 --crop-size 400 --gpu-ids 0,1 --checkname unet --eval-interval 1 --dataset rssrai2019