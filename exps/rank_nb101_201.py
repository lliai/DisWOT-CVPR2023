import json 
import argparse
import os
import random
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scipy.stats
from scipy.stats import stats
from nas_201_api import NASBench201API
from models.nasbench101.build import query_nb101_acc


def spearman(true_vector, pred_vector):
    coef, p_value = scipy.stats.spearmanr(true_vector, pred_vector)
    return coef


def calculate_nb201():        
    nb201_api = NASBench201API(
        file_path_or_dict='data/NAS-Bench-201-v1_0-e61699.pth', verbose=False)

    with open('./data/dist_nb201_bench.json', 'r') as f:
        j_file = json.load(f)

    gt_list = []
    dt_list = []

    for k, v in j_file.items():
        xinfo = nb201_api.get_more_info(int(k), dataset='cifar100', hp='200')
        gt_acc = xinfo['test-accuracy']
        dt_list.append(float(v))
        gt_list.append(gt_acc)

    print(spearman(gt_list, dt_list))

def calculate_nb101():
    with open('./data/dist_nb101_bench.json', 'r') as f:
        j_file = json.load(f)

    gt_list = []
    dt_list = []

    for k, v in j_file.items():
        gt_acc = query_nb101_acc(k)
        dt_list.append(float(v))
        gt_list.append(gt_acc)

    print(spearman(gt_list, dt_list))

calculate_nb101()