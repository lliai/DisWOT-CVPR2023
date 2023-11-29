# sample scripts for training vanilla teacher models

# python train_teacher.py --model wrn_40_2

# python train_teacher.py --model resnet56

# python train_teacher.py --model resnet110

# python train_teacher.py --model resnet32x4

# python train_teacher.py --model vgg13

# python train_teacher.py --model ResNet50


# python train_teacher.py --model resnet56 --dataset imagenet16

# python train_teacher.py --model resnet56 --dataset cifar10

CUDA_VISIBLE_DEVICES=0 python train_teacher.py --model searched_nb201_cifar_random --dataset cifar10 | tee ./save_dirs/vanilla/run1_vanilla_nb201_c10_kd_random.log && CUDA_VISIBLE_DEVICES=0 python train_teacher.py --model searched_nb201_cifar_reinforce --dataset cifar10 | tee ./save_dirs/run1_vanilla_nb201_c10_kd_reinforce.log && CUDA_VISIBLE_DEVICES=0 python train_teacher.py --model searched_nb201_cifar_zen --dataset cifar10 | tee ./save_dirs/run1_vanilla_nb201_c10_kd_zen.log

CUDA_VISIBLE_DEVICES=1 python train_teacher.py --model searched_nb201_cifar_nwot --dataset cifar10 | tee ./save_dirs/run1_vanilla_nb201_c10_kd_nwot.log && CUDA_VISIBLE_DEVICES=1 python train_teacher.py --model searched_nb201_cifar_diswot --dataset cifar10 | tee ./save_dirs/run1_vanilla_nb201_c10_kd_diswot.log && CUDA_VISIBLE_DEVICES=1 python train_teacher.py --model searched_nb201_cifar_zen --dataset cifar100 | tee ./save_dirs/run1_vanilla_nb201_c100_kd_zen.log

CUDA_VISIBLE_DEVICES=2 python train_teacher.py --model searched_nb201_cifar_nwot --dataset cifar100 | tee ./save_dirs/run1_vanilla_nb201_c100_kd_nwot.log && CUDA_VISIBLE_DEVICES=2 python train_teacher.py --model searched_nb201_cifar_diswot --dataset cifar100 | tee ./save_dirs/run1_vanilla_nb201_c100_kd_diswot.log && CUDA_VISIBLE_DEVICES=2 python train_teacher.py --model searched_nb201_cifar_zen --dataset imagenet16 | tee ./save_dirs/run1_vanilla_nb201_img16_kd_zen.log

CUDA_VISIBLE_DEVICES=1 python train_teacher.py --model searched_nb201_cifar_nwot --dataset imagenet16 | tee ./save_dirs/run1_vanilla_nb201_img16_kd_nwot.log && CUDA_VISIBLE_DEVICES=1 python train_teacher.py --model searched_nb201_cifar_diswot --dataset imagenet16 | tee ./save_dirs/run1_vanilla_nb201_img16_kd_diswot.log && CUDA_VISIBLE_DEVICES=1 python train_teacher.py --model searched_nb201_cifar_rminas --dataset imagenet16 | tee ./save_dirs/run1_vanilla_nb201_img16_kd_rminas.log
