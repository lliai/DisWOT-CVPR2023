# sample scripts for running the distillation code
# use resnet32x4 and resnet8x4 as an example

# AutoKD-Feature-alone

CUDA_VISIBLE_DEVICES=0  python /sicheng01/RepDistiller-master/RepDistiller-master/train_student.py --path_t /sicheng01/RepDistiller-master/RepDistiller-master/save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill hint --model_s resnet20 -a 0 -b 100 --trial 1|tee /sicheng01/RepDistiller-master/RepDistiller-master/logs/1-b100.log

CUDA_VISIBLE_DEVICES=0  python /sicheng01/RepDistiller-master/RepDistiller-master/train_student.py --path_t /sicheng01/RepDistiller-master/RepDistiller-master/save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill hint --model_s resnet20 -a 0 -b 100 --trial 1 --use_channel True|tee /sicheng01/RepDistiller-master/RepDistiller-master/logs/2-b100.log
CUDA_VISIBLE_DEVICES=0  python /sicheng01/RepDistiller-master/RepDistiller-master/train_student.py --path_t /sicheng01/RepDistiller-master/RepDistiller-master/save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill hint --model_s resnet20 -a 0 -b 100 --trial 1 --use_channel True --use_kl True|tee /sicheng01/RepDistiller-master/RepDistiller-master/logs/3-b100.log
CUDA_VISIBLE_DEVICES=0  python /sicheng01/RepDistiller-master/RepDistiller-master/train_student.py --path_t /sicheng01/RepDistiller-master/RepDistiller-master/save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill hint --model_s resnet20 -a 0 -b 100 --trial 1 --use_batch True|tee /sicheng01/RepDistiller-master/RepDistiller-master/logs/4-b100.log
CUDA_VISIBLE_DEVICES=0  python /sicheng01/RepDistiller-master/RepDistiller-master/train_student.py --path_t /sicheng01/RepDistiller-master/RepDistiller-master/save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill hint --model_s resnet20 -a 0 -b 100 --trial 1 --use_batch True --use_kl True|tee /sicheng01/RepDistiller-master/RepDistiller-master/logs/5-b100.log
CUDA_VISIBLE_DEVICES=0  python /sicheng01/RepDistiller-master/RepDistiller-master/train_student.py --path_t /sicheng01/RepDistiller-master/RepDistiller-master/save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill hint --model_s resnet20 -a 0 -b 100 --trial 1 --use_ms True|tee /sicheng01/RepDistiller-master/RepDistiller-master/logs/6-b100.log
CUDA_VISIBLE_DEVICES=0  python /sicheng01/RepDistiller-master/RepDistiller-master/train_student.py --path_t /sicheng01/RepDistiller-master/RepDistiller-master/save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill hint --model_s resnet20 -a 0 -b 100 --trial 1 --use_ms True --use_channel True|tee /sicheng01/RepDistiller-master/RepDistiller-master/logs/7-b100.log
CUDA_VISIBLE_DEVICES=0  python /sicheng01/RepDistiller-master/RepDistiller-master/train_student.py --path_t /sicheng01/RepDistiller-master/RepDistiller-master/save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill hint --model_s resnet20 -a 0 -b 100 --trial 1 --use_ms True --use_channel True --use_kl True|tee /sicheng01/RepDistiller-master/RepDistiller-master/logs/8-b100.log
CUDA_VISIBLE_DEVICES=0  python /sicheng01/RepDistiller-master/RepDistiller-master/train_student.py --path_t /sicheng01/RepDistiller-master/RepDistiller-master/save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill hint --model_s resnet20 -a 0 -b 100 --trial 1 --use_ms True  --use_batch True|tee /sicheng01/RepDistiller-master/RepDistiller-master/logs/9-b100.log
CUDA_VISIBLE_DEVICES=0  python /sicheng01/RepDistiller-master/RepDistiller-master/train_student.py --path_t /sicheng01/RepDistiller-master/RepDistiller-master/save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill hint --model_s resnet20 -a 0 -b 100 --trial 1 --use_ms True  --use_batch True --use_kl True|tee /sicheng01/RepDistiller-master/RepDistiller-master/logs/10-b100.log
CUDA_VISIBLE_DEVICES=0  python /sicheng01/RepDistiller-master/RepDistiller-master/train_student.py --path_t /sicheng01/RepDistiller-master/RepDistiller-master/save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill hint --model_s resnet20 -a 0 -b 100 --trial 1 --use_att True|tee /sicheng01/RepDistiller-master/RepDistiller-master/logs/11-b100.log
CUDA_VISIBLE_DEVICES=0  python /sicheng01/RepDistiller-master/RepDistiller-master/train_student.py --path_t /sicheng01/RepDistiller-master/RepDistiller-master/save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill hint --model_s resnet20 -a 0 -b 100 --trial 1 --use_channel True --use_att True|tee /sicheng01/RepDistiller-master/RepDistiller-master/logs/12-b100.log



CUDA_VISIBLE_DEVICES=0  python /sicheng01/RepDistiller-master/RepDistiller-master/train_student.py --path_t /sicheng01/RepDistiller-master/RepDistiller-master/save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill hint --model_s resnet20 -a 0 -b 100 --trial 1 --use_channel True --use_kl True --use_att True|tee /sicheng01/RepDistiller-master/RepDistiller-master/logs/13-b100.log
CUDA_VISIBLE_DEVICES=0  python /sicheng01/RepDistiller-master/RepDistiller-master/train_student.py --path_t /sicheng01/RepDistiller-master/RepDistiller-master/save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill hint --model_s resnet20 -a 0 -b 100 --trial 1 --use_batch True --use_att True|tee /sicheng01/RepDistiller-master/RepDistiller-master/logs/14-b100.log
CUDA_VISIBLE_DEVICES=0  python /sicheng01/RepDistiller-master/RepDistiller-master/train_student.py --path_t /sicheng01/RepDistiller-master/RepDistiller-master/save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill hint --model_s resnet20 -a 0 -b 100 --trial 1 --use_batch True --use_kl True --use_att True|tee /sicheng01/RepDistiller-master/RepDistiller-master/logs/15-b100.log
CUDA_VISIBLE_DEVICES=0  python /sicheng01/RepDistiller-master/RepDistiller-master/train_student.py --path_t /sicheng01/RepDistiller-master/RepDistiller-master/save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill hint --model_s resnet20 -a 0 -b 100 --trial 1 --use_ms True --use_att True|tee /sicheng01/RepDistiller-master/RepDistiller-master/logs/16-b100.log
CUDA_VISIBLE_DEVICES=0  python /sicheng01/RepDistiller-master/RepDistiller-master/train_student.py --path_t /sicheng01/RepDistiller-master/RepDistiller-master/save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill hint --model_s resnet20 -a 0 -b 100 --trial 1 --use_ms True --use_channel True --use_att True|tee /sicheng01/RepDistiller-master/RepDistiller-master/logs/17-b100.log
CUDA_VISIBLE_DEVICES=0  python /sicheng01/RepDistiller-master/RepDistiller-master/train_student.py --path_t /sicheng01/RepDistiller-master/RepDistiller-master/save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill hint --model_s resnet20 -a 0 -b 100 --trial 1 --use_ms True --use_channel True --use_kl True --use_att True|tee /sicheng01/RepDistiller-master/RepDistiller-master/logs/18-b100.log
CUDA_VISIBLE_DEVICES=0  python /sicheng01/RepDistiller-master/RepDistiller-master/train_student.py --path_t /sicheng01/RepDistiller-master/RepDistiller-master/save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill hint --model_s resnet20 -a 0 -b 100 --trial 1 --use_ms True  --use_batch True --use_att True|tee /sicheng01/RepDistiller-master/RepDistiller-master/logs/19-b100.log
CUDA_VISIBLE_DEVICES=0  python /sicheng01/RepDistiller-master/RepDistiller-master/train_student.py --path_t /sicheng01/RepDistiller-master/RepDistiller-master/save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill hint --model_s resnet20 -a 0 -b 100 --trial 1 --use_ms True  --use_batch True --use_kl True --use_att True|tee /sicheng01/RepDistiller-master/RepDistiller-master/logs/20-b100.log
CUDA_VISIBLE_DEVICES=0  python /sicheng01/RepDistiller-master/RepDistiller-master/train_student.py --path_t /sicheng01/RepDistiller-master/RepDistiller-master/save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill hint --model_s resnet20 -a 0 -b 100 --trial 1 --use_local True |tee /sicheng01/RepDistiller-master/RepDistiller-master/logs/21-b100.log
CUDA_VISIBLE_DEVICES=0  python /sicheng01/RepDistiller-master/RepDistiller-master/train_student.py --path_t /sicheng01/RepDistiller-master/RepDistiller-master/save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill hint --model_s resnet20 -a 0 -b 100 --trial 1 --use_local True --use_kl True|tee /sicheng01/RepDistiller-master/RepDistiller-master/logs/22-b100.log
CUDA_VISIBLE_DEVICES=0  python /sicheng01/RepDistiller-master/RepDistiller-master/train_student.py --path_t /sicheng01/RepDistiller-master/RepDistiller-master/save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill hint --model_s resnet20 -a 0 -b 100 --trial 1 --use_local True --use_att True|tee /sicheng01/RepDistiller-master/RepDistiller-master/logs/23-b100.log
CUDA_VISIBLE_DEVICES=0  python /sicheng01/RepDistiller-master/RepDistiller-master/train_student.py --path_t /sicheng01/RepDistiller-master/RepDistiller-master/save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill hint --model_s resnet20 -a 0 -b 100 --trial 1 --use_local True --use_kl True --use_att True|tee /sicheng01/RepDistiller-master/RepDistiller-master/logs/24-b100.log












# kd
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8x4 -r 0.1 -a 0.9 -b 0 --trial 1
# FitNet
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill hint --model_s resnet8x4 -a 0 -b 100 --trial 1
# AT
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill attention --model_s resnet8x4 -a 0 -b 1000 --trial 1
# SP
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill similarity --model_s resnet8x4 -a 0 -b 3000 --trial 1
# CC
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill correlation --model_s resnet8x4 -a 0 -b 0.02 --trial 1
# VID
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill vid --model_s resnet8x4 -a 0 -b 1 --trial 1
# RKD
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill rkd --model_s resnet8x4 -a 0 -b 1 --trial 1
# PKT
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill pkt --model_s resnet8x4 -a 0 -b 30000 --trial 1
# AB
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill abound --model_s resnet8x4 -a 0 -b 1 --trial 1
# FT
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill factor --model_s resnet8x4 -a 0 -b 200 --trial 1
# FSP
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill fsp --model_s resnet8x4 -a 0 -b 50 --trial 1
# NST
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill nst --model_s resnet8x4 -a 0 -b 50 --trial 1
# CRD
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill crd --model_s resnet8x4 -a 0 -b 0.8 --trial 1

# CRD+KD
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill crd --model_s resnet8x4 -a 1 -b 0.8 --trial 1
