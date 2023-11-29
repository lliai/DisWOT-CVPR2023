#!/bin/bash

module load anaconda
module load cuda/11.2
module load cudnn/8.1.0.77_CUDA11.2
source activate py38

mkdir -p save_dirs

# CUDA_VISIBLE_DEVICES=0 python train_nb201_student.py --path_t ./save/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --dataset cifar100 -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 --epochs 90 --arch_index 1 | tee ./save_dirs/arch_idx_1.log


# CUDA_VISIBLE_DEVICES=0 python train_nb201_student.py --path_t ./save/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --dataset cifar100 -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 --epochs 90 --arch_index 10 | tee ./save_dirs/arch_idx_10.log


# CUDA_VISIBLE_DEVICES=0 python train_nb201_student.py --path_t ./save/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --dataset cifar100 -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 --epochs 90 --arch_index 20 | tee ./save_dirs/arch_idx_20.log



# CUDA_VISIBLE_DEVICES=0 python train_nb201_student.py --path_t ./save/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --dataset cifar100 -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 --epochs 90 --arch_index 40 | tee ./save_dirs/arch_idx_40.log



# CUDA_VISIBLE_DEVICES=0 python train_nb201_student.py --path_t ./save/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --dataset cifar100 -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 --epochs 90 --arch_index 80 | tee ./save_dirs/arch_idx_80.log



# CUDA_VISIBLE_DEVICES=0 python train_nb201_student.py --path_t ./save/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --dataset cifar100 -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 --epochs 90 --arch_index 120 | tee ./save_dirs/arch_idx_120.log



# CUDA_VISIBLE_DEVICES=0 python train_nb201_student.py --path_t ./save/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --dataset cifar100 -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 --epochs 90 --arch_index 240 | tee ./save_dirs/arch_idx_240.log



# CUDA_VISIBLE_DEVICES=0 python train_nb201_student.py --path_t ./save/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --dataset cifar100 -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 --epochs 90 --arch_index 360 | tee ./save_dirs/arch_idx_360.log



# CUDA_VISIBLE_DEVICES=0 python train_nb201_student.py --path_t ./save/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --dataset cifar100 -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 --epochs 90 --arch_index 480 | tee ./save_dirs/arch_idx_480.log



# CUDA_VISIBLE_DEVICES=0 python train_nb201_student.py --path_t ./save/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --dataset cifar100 -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 --epochs 90 --arch_index 520 | tee ./save_dirs/arch_idx_520.log



# CUDA_VISIBLE_DEVICES=0 python train_nb201_student.py --path_t ./save/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --dataset cifar100 -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 --epochs 90 --arch_index 1024 | tee ./save_dirs/arch_idx_1024.log




# CUDA_VISIBLE_DEVICES=0 python train_nb201_student.py --path_t ./save/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --dataset cifar100 -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 --epochs 90 --arch_index 2096 | tee ./save_dirs/arch_idx_2096.log




CUDA_VISIBLE_DEVICES=0 python train_nb201_student.py --path_t ./save/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --dataset cifar100 -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 --epochs 90 --arch_index 3069 | tee ./save_dirs/arch_idx_3069.log && CUDA_VISIBLE_DEVICES=0 python train_nb201_student.py --path_t ./save/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --dataset cifar100 -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 --epochs 90 --arch_index 4058 | tee ./save_dirs/arch_idx_4058.log




CUDA_VISIBLE_DEVICES=0 python train_nb201_student.py --path_t ./save/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --dataset cifar100 -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 --epochs 90 --arch_index 5066 | tee ./save_dirs/arch_idx_5066.log

