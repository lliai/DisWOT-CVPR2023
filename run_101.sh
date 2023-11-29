#!/bin/bash

module load anaconda
module load cuda/11.2
module load cudnn/8.1.0.77_CUDA11.2
source activate py38

echo "run 101 distill kd zc [ICKD]"
python exps/parse_nb101_distill.py
