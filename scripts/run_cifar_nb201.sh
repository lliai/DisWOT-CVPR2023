#!/bin/bash


########### ImageNet16

# CUDA_VISIBLE_DEVICES=0 python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --dataset imagenet16 --model_s searched_nb201_cifar_diswot -r 0.1 -a 0.9 -b 0 --trial 1 --batch_size 128 --learning_rate 0.1 | tee ./save_dirs/run1_cifar_nb201_img16_kd_diswot.log &&  CUDA_VISIBLE_DEVICES=0 python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --dataset imagenet16 --model_s searched_nb201_cifar_diswot -r 0.1 -a 0.9 -b 0 --trial 2 --batch_size 128 --learning_rate 0.1 | tee ./save_dirs/run2_cifar_nb201_img16_kd_diswot.log



# CUDA_VISIBLE_DEVICES=1 python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --dataset imagenet16 --model_s searched_nb201_cifar_rminas -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 128 --learning_rate 0.1 | tee ./save_dirs/run1_cifar_nb201_img16_kd_rminas.log &&  CUDA_VISIBLE_DEVICES=1 python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --dataset imagenet16 --model_s searched_nb201_cifar_rminas -r 0.1 -a 0.9 -b 0  --trial 2  --batch_size 128 --learning_rate 0.1 | tee ./save_dirs/run2_cifar_nb201_img16_kd_rminas.log


# CUDA_VISIBLE_DEVICES=4 python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --dataset imagenet16 --model_s searched_nb201_cifar_enas -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 64 --learning_rate 0.1 | tee ./save_dirs/run1_cifar_nb201_img16_kd_enas.log && CUDA_VISIBLE_DEVICES=4 python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --dataset imagenet16 --model_s searched_nb201_cifar_enas -r 0.1 -a 0.9 -b 0  --trial 2  --batch_size 64 --learning_rate 0.1 | tee ./save_dirs/run2_cifar_nb201_img16_kd_enas.log



# CUDA_VISIBLE_DEVICES=3 python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --dataset imagenet16 --model_s searched_nb201_cifar_gdas -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 128 --learning_rate 0.1 | tee ./save_dirs/run1_cifar_nb201_img16_kd_gdas.log &&  CUDA_VISIBLE_DEVICES=3 python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --dataset imagenet16 --model_s searched_nb201_cifar_gdas -r 0.1 -a 0.9 -b 0  --trial 2  --batch_size 128 --learning_rate 0.1 | tee ./save_dirs/run2_cifar_nb201_img16_kd_gdas.log



# CUDA_VISIBLE_DEVICES=4 python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --dataset imagenet16 --model_s searched_nb201_cifar_darts -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 128 --learning_rate 0.1 | tee ./save_dirs/run1_cifar_nb201_img16_kd_darts.log && CUDA_VISIBLE_DEVICES=4 python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --dataset imagenet16 --model_s searched_nb201_cifar_darts -r 0.1 -a 0.9 -b 0  --trial 2  --batch_size 128 --learning_rate 0.1 | tee ./save_dirs/run2_cifar_nb201_img16_kd_darts.log


# CUDA_VISIBLE_DEVICES=5 python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --dataset imagenet16 --model_s searched_nb201_cifar_setn -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 64 --learning_rate 0.05 | tee ./save_dirs/run1_cifar_nb201_img16_kd_setn.log && CUDA_VISIBLE_DEVICES=5 python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --dataset imagenet16 --model_s searched_nb201_cifar_setn -r 0.1 -a 0.9 -b 0  --trial 2  --batch_size 64 --learning_rate 0.05 | tee ./save_dirs/run2_cifar_nb201_img16_kd_setn.log


# CUDA_VISIBLE_DEVICES=6 python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --dataset imagenet16 --model_s searched_nb201_cifar_ea -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 128 --learning_rate 0.1 | tee ./save_dirs/run1_cifar_nb201_img16_kd_ea.log && CUDA_VISIBLE_DEVICES=6 python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --dataset imagenet16 --model_s searched_nb201_cifar_ea -r 0.1 -a 0.9 -b 0  --trial 2  --batch_size 128 --learning_rate 0.1 | tee ./save_dirs/run2_cifar_nb201_img16_kd_ea.log

# CUDA_VISIBLE_DEVICES=7 python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --dataset imagenet16 --model_s searched_nb201_cifar_random -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 128 --learning_rate 0.1 | tee ./save_dirs/run1_cifar_nb201_img16_kd_random.log && CUDA_VISIBLE_DEVICES=7 python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --dataset imagenet16 --model_s searched_nb201_cifar_random -r 0.1 -a 0.9 -b 0  --trial 2  --batch_size 128 --learning_rate 0.1 | tee ./save_dirs/run2_cifar_nb201_img16_kd_random.log

# ------------------------------------------------------------------------------------------------------------------------next page-------------------------------


# CUDA_VISIBLE_DEVICES=1 python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --dataset imagenet16 --model_s searched_nb201_cifar_reinforce -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 128 --learning_rate 0.1 | tee ./save_dirs/run1_cifar_nb201_img16_kd_reinforce.log && CUDA_VISIBLE_DEVICES=8 python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --dataset imagenet16 --model_s searched_nb201_cifar_reinforce -r 0.1 -a 0.9 -b 0  --trial 2  --batch_size 128 --learning_rate 0.1 | tee ./save_dirs/run2_cifar_nb201_img16_kd_reinforce.log

# CUDA_VISIBLE_DEVICES=3 python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --dataset imagenet16 --model_s searched_nb201_cifar_rnasd -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 128 --learning_rate 0.1 | tee ./save_dirs/run1_cifar_nb201_img16_kd_rnasd.log && CUDA_VISIBLE_DEVICES=0 python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --dataset imagenet16 --model_s searched_nb201_cifar_rnasd -r 0.1 -a 0.9 -b 0  --trial 2  --batch_size 128 --learning_rate 0.1 | tee ./save_dirs/run2_cifar_nb201_img16_kd_rnasd.log

# CUDA_VISIBLE_DEVICES=5 python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --dataset imagenet16 --model_s searched_nb201_cifar_rdnas -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 128 --learning_rate 0.1 | tee ./save_dirs/run1_cifar_nb201_img16_kd_rdnas.log && CUDA_VISIBLE_DEVICES=1 python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --dataset imagenet16 --model_s searched_nb201_cifar_rdnas -r 0.1 -a 0.9 -b 0  --trial 2  --batch_size 128 --learning_rate 0.1 | tee ./save_dirs/run2_cifar_nb201_img16_kd_rdnas.log

# CUDA_VISIBLE_DEVICES=6 python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --dataset imagenet16 --model_s searched_nb201_cifar_spos -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 128 --learning_rate 0.1 | tee ./save_dirs/run1_cifar_nb201_img16_kd_spos.log && CUDA_VISIBLE_DEVICES=3 python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --dataset imagenet16 --model_s searched_nb201_cifar_spos -r 0.1 -a 0.9 -b 0  --trial 2  --batch_size 128 --learning_rate 0.1 | tee ./save_dirs/run2_cifar_nb201_img16_kd_spos.log

## -----------------rerun the failed model



########### CIFAR10 3x3090Ti


# CUDA_VISIBLE_DEVICES=0 python train_student.py --path_t ./save/c10_models/base-c10-r110/model_best.pth.tar --distill kd --dataset cifar10 --model_s searched_nb201_cifar_diswot -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 | tee ./save_dirs/run1_cifar10_nb201_c10_kd_diswot.log && CUDA_VISIBLE_DEVICES=1 python train_student.py --path_t ./save/c10_models/base-c10-r110/model_best.pth.tar --distill kd --dataset cifar10 --model_s searched_nb201_cifar_rminas -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 | tee ./save_dirs/run1_cifar10_nb201_c10_kd_rminas.log && CUDA_VISIBLE_DEVICES=2 python train_student.py --path_t ./save/c10_models/base-c10-r110/model_best.pth.tar --distill kd --dataset cifar10 --model_s searched_nb201_cifar_enas -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 | tee ./save_dirs/run1_cifar10_nb201_c10_kd_enas.log && CUDA_VISIBLE_DEVICES=0 python train_student.py --path_t ./save/c10_models/base-c10-r110/model_best.pth.tar --distill kd --dataset cifar10 --model_s searched_nb201_cifar_gdas -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 | tee ./save_dirs/run1_cifar10_nb201_c10_kd_gdas.log

# CUDA_VISIBLE_DEVICES=1 python train_student.py --path_t ./save/c10_models/base-c10-r110/model_best.pth.tar --distill kd --dataset cifar10 --model_s searched_nb201_cifar_darts -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 | tee ./save_dirs/run1_cifar10_nb201_c10_kd_darts.log && CUDA_VISIBLE_DEVICES=2 python train_student.py --path_t ./save/c10_models/base-c10-r110/model_best.pth.tar --distill kd --dataset cifar10 --model_s searched_nb201_cifar_setn -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 | tee ./save_dirs/run1_cifar10_nb201_c10_kd_setn.log && CUDA_VISIBLE_DEVICES=0 python train_student.py --path_t ./save/c10_models/base-c10-r110/model_best.pth.tar --distill kd --dataset cifar10 --model_s searched_nb201_cifar_ea -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 | tee ./save_dirs/run1_cifar10_nb201_c10_kd_ea.log && CUDA_VISIBLE_DEVICES=1 python train_student.py --path_t ./save/c10_models/base-c10-r110/model_best.pth.tar --distill kd --dataset cifar10 --model_s searched_nb201_cifar_random -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 | tee ./save_dirs/run1_cifar10_nb201_c10_kd_random.log

# CUDA_VISIBLE_DEVICES=2 python train_student.py --path_t ./save/c10_models/base-c10-r110/model_best.pth.tar --distill kd --dataset cifar10 --model_s searched_nb201_cifar_reinforce -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 | tee ./save_dirs/run1_cifar10_nb201_c10_kd_reinforce.log && CUDA_VISIBLE_DEVICES=0 python train_student.py --path_t ./save/c10_models/base-c10-r110/model_best.pth.tar --distill kd --dataset cifar10 --model_s searched_nb201_cifar_rnasd -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 | tee ./save_dirs/run1_cifar10_nb201_c10_kd_rnasd.log && CUDA_VISIBLE_DEVICES=1 python train_student.py --path_t ./save/c10_models/base-c10-r110/model_best.pth.tar --distill kd --dataset cifar10 --model_s searched_nb201_cifar_rdnas -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 | tee ./save_dirs/run1_cifar10_nb201_c10_kd_rdnas.log

# CUDA_VISIBLE_DEVICES=2 python train_student.py --path_t ./save/c10_models/base-c10-r110/model_best.pth.tar --distill kd --dataset cifar10 --model_s searched_nb201_cifar_spos -r 0.1 -a 0.9 -b 0  --trial 3 --batch_size 128 --learning_rate 0.1 | tee ./save_dirs/run3_cifar10_nb201_c10_kd_spos.log && CUDA_VISIBLE_DEVICES=2 python train_student.py --path_t ./save/c10_models/base-c10-r110/model_best.pth.tar --distill kd --dataset cifar10 --model_s searched_nb201_cifar_spos -r 0.1 -a 0.9 -b 0  --trial 3  --batch_size 128 --learning_rate 0.1 | tee ./save_dirs/run3_cifar10_nb201_c10_kd_spos.log

# # 1 3 5 6


### ----------------------------- zero cost proxy ----------------------------------------------------------

# cifar10
CUDA_VISIBLE_DEVICES=1 python train_student.py --path_t ./save/c10_models/base-c10-r110/model_best.pth.tar --distill kd --dataset cifar10 --model_s searched_nb201_cifar_nwot -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 | tee ./save_dirs/run1_cifar10_nb201_c10_kd_nwot.log && CUDA_VISIBLE_DEVICES=1 python train_student.py --path_t ./save/c10_models/base-c10-r110/model_best.pth.tar --distill kd --dataset cifar10 --model_s searched_nb201_cifar_zen -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 | tee ./save_dirs/run1_cifar10_nb201_c10_kd_zen.log


# imagenet16
CUDA_VISIBLE_DEVICES=0 python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --dataset imagenet16 --model_s searched_nb201_cifar_nwot -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 | tee ./save_dirs/run1_cifar_nb201_img16_kd_nwot.log && CUDA_VISIBLE_DEVICES=0 python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --dataset imagenet16 --model_s searched_nb201_cifar_zen -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 | tee ./save_dirs/run1_cifar_nb201_img16_kd_zen.log


### ---------rerun
CUDA_VISIBLE_DEVICES=0 python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --dataset imagenet16 --model_s searched_nb201_cifar_diswot -r 0.1 -a 0.9 -b 0 --trial 4 --batch_size 128 --learning_rate 0.1 | tee ./save_dirs/run4_cifar_nb201_img16_kd_diswot.log &&  CUDA_VISIBLE_DEVICES=0 python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --dataset imagenet16 --model_s searched_nb201_cifar_diswot -r 0.1 -a 0.9 -b 0 --trial 5 --batch_size 128 --learning_rate 0.1 | tee ./save_dirs/run5_cifar_nb201_img16_kd_diswot.log

CUDA_VISIBLE_DEVICES=1 python train_student.py --path_t ./save/c10_models/base-c10-r110/model_best.pth.tar --distill kd --dataset cifar10 --model_s searched_nb201_cifar_diswot -r 0.1 -a 0.9 -b 0  --trial 3  --batch_size 256 --learning_rate 0.1 | tee ./save_dirs/run3_cifar10_nb201_c10_kd_diswot.log && CUDA_VISIBLE_DEVICES=1 python train_student.py --path_t ./save/c10_models/base-c10-r110/model_best.pth.tar --distill kd --dataset cifar10 --model_s searched_nb201_cifar_diswot -r 0.1 -a 0.9 -b 0  --trial 4  --batch_size 256 --learning_rate 0.1 | tee ./save_dirs/run4_cifar10_nb201_c10_kd_diswot.log


## three datasets

# cifar10
# searched_nb201_cifar_diswot_ss

CUDA_VISIBLE_DEVICES=0 python train_student.py --path_t ./save/c10_models/base-c10-r110/model_best.pth.tar --distill kd --dataset cifar10 --model_s searched_nb201_cifar_diswot_ss -r 0.1 -a 0.9 -b 0  --trial 1 | tee ./save_dirs/run_last_cifar10_nb201_c10_kd_diswot_ss.log

# cifar100
# searched_nb201_cifar_diswot_ss
