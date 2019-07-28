#!/usr/bin/env bash

# experiment 0
# CUDA_VISIBLE_DEVICES=0,1 python train.py --lr 0.01 --weight-decay 0.001 --epochs 200 --batch-size 32 --test-batch-size 32 --base-size 400 --crop-size 400 --gpu-ids 0,1 --checkname unet --eval-interval 1 --dataset rssrai2019

# experiment 1  using Adam no weight-decay
CUDA_VISIBLE_DEVICES=0,1 python train.py --learn-rate 0.001 --weight-decay 0 --epochs 1000 --batch-size 32 --test-batch-size 32 --base-size 400 --crop-size 400 --gpu-ids 0,1 --checkname unet --eval-interval 1 --dataset rssrai2019

# experiment 2  add weight-decay RandomGammaTransform RandomNoise
CUDA_VISIBLE_DEVICES=0,1 python train.py --learn-rate 0.001 --weight-decay 0.001 --epochs 600 --batch-size 32 --test-batch-size 32 --base-size 400 --crop-size 400 --gpu-ids 0,1 --checkname unet --eval-interval 1 --dataset rssrai2019