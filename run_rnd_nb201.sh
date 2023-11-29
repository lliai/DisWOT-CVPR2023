#!/bin/bash

module load anaconda
module load cuda/11.2
module load cudnn/8.1.0.77_CUDA11.2
source activate py38


start=`date +%s`

CUDA_VISIBLE_DEVICES=0 python searcher/rand_search_nb201.py --iterations 200 | tee ./save_dirs/rand_ss_nb201_zc_vanillazc_try0.log && CUDA_VISIBLE_DEVICES=0 python searcher/rand_search_nb201.py --iterations 200 | tee ./save_dirs/rand_ss_nb201_zc_vanillazc_try1.log && CUDA_VISIBLE_DEVICES=0 python searcher/rand_search_nb201.py --iterations 200 | tee ./save_dirs/rand_ss_nb201_zc_vanillazc_try2.log 

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
