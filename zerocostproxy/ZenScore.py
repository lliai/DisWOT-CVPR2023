'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import time

import numpy as np
import torch
from torch import nn


def network_weight_gaussian_init(net: nn.Module):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                continue

    return net


def compute_zen_score(gpu,
                      model,
                      mixup_gamma,
                      resolution,
                      batch_size,
                      repeat,
                      fp16=False):
    info = {}
    nas_score_list = []
    if gpu is not None:
        device = torch.device('cuda:{}'.format(gpu))
    else:
        device = torch.device('cpu')

    if fp16:
        dtype = torch.half
    else:
        dtype = torch.float32

    with torch.no_grad():
        for repeat_count in range(repeat):
            network_weight_gaussian_init(model)
            input = torch.randn(size=[batch_size, 3, resolution, resolution],
                                device=device,
                                dtype=dtype)
            input2 = torch.randn(size=[batch_size, 3, resolution, resolution],
                                 device=device,
                                 dtype=dtype)
            mixup_input = input + mixup_gamma * input2
            output = model.forward_before_global_avg_pool(input)
            mixup_output = model.forward_before_global_avg_pool(mixup_input)

            nas_score = torch.sum(torch.abs(output - mixup_output),
                                  dim=[1, 2, 3])
            nas_score = torch.mean(nas_score)

            # compute BN scaling
            log_bn_scaling_factor = 0.0
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    bn_scaling_factor = torch.sqrt(torch.mean(m.running_var))
                    log_bn_scaling_factor += torch.log(bn_scaling_factor)
                pass
            pass
            nas_score = torch.log(nas_score) + log_bn_scaling_factor
            nas_score_list.append(float(nas_score))

    std_nas_score = np.std(nas_score_list)
    avg_precision = 1.96 * std_nas_score / np.sqrt(len(nas_score_list))
    avg_nas_score = np.mean(nas_score_list)

    info['avg_nas_score'] = float(avg_nas_score)
    info['std_nas_score'] = float(std_nas_score)
    info['avg_precision'] = float(avg_precision)
    return info


def parse_cmd_options(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='number of instances in one mini-batch.')
    parser.add_argument(
        '--input_image_size',
        type=int,
        default=None,
        help=
        'resolution of input image, usually 32 for CIFAR and 224 for ImageNet.'
    )
    parser.add_argument('--repeat_times', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--mixup_gamma', type=float, default=1e-2)
    module_opt, _ = parser.parse_known_args(argv)
    return module_opt
