#!/usr/bin/env bash

# experiment 0
# CUDA_VISIBLE_DEVICES=0,1 python train.py --lr 0.01 --weight-decay 0.001 --epochs 200 --batch-size 32 --test-batch-size 32 --base-size 400 --crop-size 400 --gpu-ids 0,1 --checkname unet --eval-interval 1 --dataset rssrai2019

# experiment 1  using Adam no weight-decay
# CUDA_VISIBLE_DEVICES=0,1 python train.py --learn-rate 0.001 --weight-decay 0 --epochs 1000 --batch-size 32 --test-batch-size 32 --base-size 400 --crop-size 400 --gpu-ids 0,1 --checkname unet --eval-interval 1 --dataset rssrai2019

# experiment 2  将编码器模块中每块增加一层卷积层，并在下采样层最后添加dropout=0.5, RandomGammaTransform, RandomBilateralFilter, RandomNoise
# CUDA_VISIBLE_DEVICES=0,1 python train.py --learn-rate 0.001 --weight-decay 0 --epochs 1000 --batch-size 20 --test-batch-size 20 --base-size 400 --crop-size 400 --gpu-ids 0,1 --checkname unet --eval-interval 1 --dataset rssrai2019

# experiment 3  在实验2的基础上add weight-decay amsgrad,  当epoch为总数的50%,80%时，将学习率缩小10倍
# CUDA_VISIBLE_DEVICES=0,1 python train.py --learn-rate 0.001 --weight-decay 1e-4 --epochs 1000 --batch-size 20 --test-batch-size 20 --base-size 400 --crop-size 400 --gpu-ids 0,1 --checkname unet --eval-interval 1 --dataset rssrai2019

# experiment 4  实验3 L2系数过大导致精度过低，将其继续缩小10倍。学习率更新改为每次0.3倍，增加epoch数量
# CUDA_VISIBLE_DEVICES=0,1 python train.py --learn-rate 0.001 --weight-decay 1e-5 --epochs 1500 --batch-size 20 --test-batch-size 20 --base-size 400 --crop-size 400 --gpu-ids 0,1 --checkname unet --eval-interval 1 --dataset rssrai2019

# experiment 5  使用UNetNested网络训练
CUDA_VISIBLE_DEVICES=0,1 python train.py --learn-rate 0.001 --weight-decay 0 --epochs 1000 --batch-size 12 --test-batch-size 12 --base-size 400 --crop-size 400 --gpu-ids 0,1 --backbone unetNested --checkname unetNested --eval-interval 1 --dataset rssrai2019

