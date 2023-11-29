#!/bin/bash

module load anaconda
module load cuda/11.2
module load cudnn/8.1.0.77_CUDA11.2
source activate py38


# echo "run nb101 distill -> NST"
# python exps/parse_nb101_distill.py

# echo "run nb101 vanilla -> NST"
# python exps/parse_nb101_vanilla.py


# echo "run nb201 distill -> NST"
# python exps/parse_nb201_distill.py


echo "run nb201 vanilla -> NST"
python exps/parse_nb201_vanilla.py


