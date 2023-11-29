#!/bin/bash

module load anaconda
module load cuda/11.2
module load cudnn/8.1.0.77_CUDA11.2
source activate py38

echo "@run 201 distill kd zc [SP+ICKD]"
python exps/parse_nb201_distill.py

echo "@run 201 vanilla kd zc [SP+ICKD]"
python exps/parse_nb201_vanilla.py