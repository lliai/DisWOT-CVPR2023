'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
The implementation of NASWOT score is modified from:
https://github.com/BayesWatch/nas-without-training
'''

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import time

import numpy as np
import torch
from torch import nn

from models.candidates.mutable import basic_blocks


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


def logdet(K):
    s, ld = np.linalg.slogdet(K)
    return ld


def get_batch_jacobian(net, x):
    net.zero_grad()
    x.requires_grad_(True)
    y = net(x)
    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()
    # return jacob, target.detach(), y.detach()
    return jacob, y.detach()


def compute_naswot_score(gpu, model, resolution, batch_size):
    if gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)

    network_weight_gaussian_init(model)
    input = torch.randn(size=[batch_size, 3, resolution, resolution])
    if gpu is not None:
        input = input.cuda(gpu)

    model.K = np.zeros((batch_size, batch_size))

    def counting_forward_hook(module, inp, out):
        try:
            if not module.visited_backwards:
                return
            if isinstance(inp, tuple):
                inp = inp[0]
            inp = inp.view(inp.size(0), -1)
            x = (inp > 0).float()
            K = x @ x.t()
            K2 = (1. - x) @ (1. - x.t())
            model.K = model.K + K.cpu().numpy() + K2.cpu().numpy()
        except Exception as err:
            print('---- error on model : ')
            print(model)
            raise err

    def counting_backward_hook(module, inp, out):
        module.visited_backwards = True

    for name, module in model.named_modules():
        # if 'ReLU' in str(type(module)):
        if isinstance(module, basic_blocks.RELU):
            # hooks[name] = module.register_forward_hook(counting_hook)
            module.visited_backwards = True
            module.register_forward_hook(counting_forward_hook)
            module.register_backward_hook(counting_backward_hook)

    x = input
    jacobs, y = get_batch_jacobian(model, x)

    score = logdet(model.K)

    return float(score)


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
    module_opt, _ = parser.parse_known_args(argv)
    return module_opt
