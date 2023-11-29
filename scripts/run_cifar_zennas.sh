#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill kd --model_s zen_cifar_res32_05M -r 0.1 -a 0.9 -b 0 --trial 1

CUDA_VISIBLE_DEVICES=1 python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill kd --model_s zen_cifar_res32_1M -r 0.1 -a 0.9 -b 0 --trial 1

CUDA_VISIBLE_DEVICES=2 python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill kd --model_s zen_cifar_res32_2M -r 0.1 -a 0.9 -b 0 --trial 1
