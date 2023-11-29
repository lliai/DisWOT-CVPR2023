#!/bin/bash

module load anaconda
module load cuda/11.2
module load cudnn/8.1.0.77_CUDA11.2
source activate py38

mkdir -p save_dirs

# CUDA_VISIBLE_DEVICES=0 python train_nb101_student.py --path_t ./save/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --dataset cifar100 -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 --epochs 90 --arch_hash 9bf6f79e80397de4f50e1ece2a89a26a | tee ./save_dirs/9bf6f79e80397de4f50e1ece2a89a26a.log


# CUDA_VISIBLE_DEVICES=0 python train_nb101_student.py --path_t ./save/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --dataset cifar100 -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 --epochs 90 --arch_hash 0aa07f2ad9b17b4382ffb0071126a0ab | tee ./save_dirs/0aa07f2ad9b17b4382ffb0071126a0ab.log


# CUDA_VISIBLE_DEVICES=0 python train_nb101_student.py --path_t ./save/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --dataset cifar100 -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 --epochs 90 --arch_hash 2d49b28ff1ec41076435c1358751a280 | tee ./save_dirs/2d49b28ff1ec41076435c1358751a280.log



# CUDA_VISIBLE_DEVICES=0 python train_nb101_student.py --path_t ./save/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --dataset cifar100 -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 --epochs 90 --arch_hash 3756ce086e95c91e631f73ae21f9b104 | tee ./save_dirs/3756ce086e95c91e631f73ae21f9b104.log



# CUDA_VISIBLE_DEVICES=0 python train_nb101_student.py --path_t ./save/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --dataset cifar100 -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 --epochs 90 --arch_hash ed0d6606073574ab143c34de4d8fd912 | tee ./save_dirs/ed0d6606073574ab143c34de4d8fd912.log


# CUDA_VISIBLE_DEVICES=0 python train_nb101_student.py --path_t ./save/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --dataset cifar100 -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 --epochs 90 --arch_hash 838d107a40f4b32f8cd2fc1510709903 | tee ./save_dirs/838d107a40f4b32f8cd2fc1510709903.log


# CUDA_VISIBLE_DEVICES=0 python train_nb101_student.py --path_t ./save/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --dataset cifar100 -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 --epochs 90 --arch_hash 2c7959a7c86a2c59ebeb7cc80c7c158d | tee ./save_dirs/2c7959a7c86a2c59ebeb7cc80c7c158d.log


# CUDA_VISIBLE_DEVICES=0 python train_nb101_student.py --path_t ./save/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --dataset cifar100 -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 --epochs 90 --arch_hash 4a6b2f259c83cb4d32b543376d60d432 | tee ./save_dirs/4a6b2f259c83cb4d32b543376d60d432.log


# CUDA_VISIBLE_DEVICES=0 python train_nb101_student.py --path_t ./save/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --dataset cifar100 -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 --epochs 90 --arch_hash 271e604672eb93b263a0a91117bc11b3 | tee ./save_dirs/271e604672eb93b263a0a91117bc11b3.log


# CUDA_VISIBLE_DEVICES=0 python train_nb101_student.py --path_t ./save/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --dataset cifar100 -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 --epochs 90 --arch_hash 1ff97e7a5f5b02a199d20ad35b310783 | tee ./save_dirs/1ff97e7a5f5b02a199d20ad35b310783.log


# CUDA_VISIBLE_DEVICES=0 python train_nb101_student.py --path_t ./save/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --dataset cifar100 -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 --epochs 90 --arch_hash 19cf5058e237bdd06e270f93e205407c | tee ./save_dirs/19cf5058e237bdd06e270f93e205407c.log


# CUDA_VISIBLE_DEVICES=0 python train_nb101_student.py --path_t ./save/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --dataset cifar100 -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 --epochs 90 --arch_hash d123436366642a9aed1116e5b0e0157a | tee ./save_dirs/d123436366642a9aed1116e5b0e0157a.log



CUDA_VISIBLE_DEVICES=0 python train_nb101_student.py --path_t ./save/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --dataset cifar100 -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 --epochs 90 --arch_hash 5cdc1659ea10013a21350317f919013d | tee ./save_dirs/5cdc1659ea10013a21350317f919013d.log


CUDA_VISIBLE_DEVICES=0 python train_nb101_student.py --path_t ./save/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --dataset cifar100 -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 --epochs 90 --arch_hash f03cd1b16f985e15f2144adf7c91c35e | tee ./save_dirs/f03cd1b16f985e15f2144adf7c91c35e.log


CUDA_VISIBLE_DEVICES=0 python train_nb101_student.py --path_t ./save/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --dataset cifar100 -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 --epochs 90 --arch_hash 0ddbd5b26c6b0fbeabf9e43374100698 | tee ./save_dirs/0ddbd5b26c6b0fbeabf9e43374100698.log