#!/bin/bash
# searched_nb201_cifar_diswot_sp with kd distillation.


# cifar10
CUDA_VISIBLE_DEVICES=0 python train_student.py --path_t ./save/c10_models/base-c10-r110/model_best.pth.tar \
     --distill kd --dataset cifar10 \
     --model_s searched_nb201_cifar_diswot_ss \
     -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 64 \
     --learning_rate 0.05 | tee ./save_dirs/run_diswot_sp/run_last_cifar10_nb201_kd_diswot_ss.log

# cifar100
CUDA_VISIBLE_DEVICES=1 python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth \
     --distill kd --dataset cifar100 \
     --model_s searched_nb201_cifar_diswot_ss \
     -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 64 \
     --learning_rate 0.05 | tee ./save_dirs/run_diswot_sp/run_last_cifar100_nb201_kd_diswot_ss.log

# imagenet16
CUDA_VISIBLE_DEVICES=2 python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth \
     --distill kd --dataset imagenet16 \
     --model_s searched_nb201_cifar_diswot_ss \
     -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 \
     --learning_rate 0.1 | tee ./save_dirs/run_diswot_sp/run_last_img16_nb201_kd_diswot_ss.log
