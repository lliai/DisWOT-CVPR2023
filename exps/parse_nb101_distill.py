import argparse
import json
import os
import time 
import sys
from typing import List
import numpy as np 
from numpy import ndarray 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset.cifar10 import get_cifar10_dataloaders
from dataset.cifar100 import get_cifar100_dataloaders
from dataset.imagenet16 import get_imagenet16_dataloaders
from distiller_zoo import ICKDLoss, RMIloss, Similarity, build_loss
from models.nasbench101.build import (S50_v0, S50_v1, S50_v2,
                                      get_nb101_model_and_acc,
                                      get_nb101_teacher)
from predictor.pruners import predictive
from rank_consisteny import kendalltau, pearson, spearman



def network_weight_gaussian_init(net: nn.Module):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is None:
                    continue
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                continue
    return net


def compute_vanilla_distiller_rank(dataloader,
                                   loss_name,
                                   j_file,
                                   TARGET='cifar10',
                                   NUM_CLASSES=10):
    """only compute the rank of distiller loss
    """
    tnet = get_nb101_teacher()
    # network_weight_gaussian_init(tnet)

    dataiter = iter(dataloader)
    img, label = next(dataiter)


    if torch.cuda.is_available():
        tnet = tnet.to('cuda:0')
        img, label = img.to('cuda:0'), label.to('cuda:0')

    # build criterion
    if loss_name != 'RMI':
        criterion = build_loss(loss_name)
    else:
        criterion = RMIloss(img.size()[0])

    # record gt and zc results
    gt_list = []
    zcs_list = []

    # begin computation
    for arch_hash, dacc in j_file.items():
        # get random subnet
        snet, acc = get_nb101_model_and_acc(arch_hash)
        # network_weight_gaussian_init(snet)

        gt_list.append(dacc)

        try:
            img, label = next(dataiter)
        except StopIteration as e:
            dataiter = iter(dataloader)
            img, label = next(dataiter)

        if torch.cuda.is_available():
            snet = snet.to('cuda:0')
            img, label = img.to('cuda:0'), label.to('cuda:0')

        tfeature, tlogits = tnet.forward_with_features(img)
        sfeature, slogits = snet.forward_with_features(img)

        # criterion_ce(tlogits, label).backward()
        # criterion_ce(slogits, label).backward()
        # tcompressed = tnet.classifier.weight.grad.unsqueeze(-1).unsqueeze(-1)
        # scompressed = snet.classifier.weight.grad.unsqueeze(-1).unsqueeze(-1)
        # visulize_ickd_G(tcompressed.detach(), scompressed.detach(), name=f'fig_of_{i}_th_subnet')

        if loss_name == 'KD':
            result = -1 * criterion(tlogits, slogits)
        elif loss_name in ['FitNet', 'RKD', 'PKT', 'CC']:
            result = -1 * \
                criterion(tfeature[-1], sfeature[-1])
        elif loss_name in ['AT', 'NST', 'KDSVD']:
            result = -1 * \
                sum(criterion(tfeature[1:-1], sfeature[1:-1]))
        elif loss_name == 'SP':
            result = -1 * \
                criterion(tfeature[-2], sfeature[-2])[0]
        else:
            raise f'Not support {loss_name} currently.'
        
        if torch.cuda.is_available():
            result = result.cpu().detach().numpy()
        else:
            result = result.detach().numpy()

        if isinstance(result, (list, tuple)):
            result = result[0]
        elif isinstance(result, ndarray):
            result = result.item()
        else:
            raise NotImplementedError

        zcs_list.append(result if isinstance(result, float) else result[0])

    print(
        f'{loss_name} kd: {kendalltau(zcs_list,gt_list):.4f} sp: {spearman(zcs_list,gt_list):.4f} ps: {pearson(zcs_list, gt_list):.4f}'
    )

    return kendalltau(zcs_list,
                      gt_list), spearman(zcs_list,
                                         gt_list), pearson(zcs_list, gt_list)


def compute_diswot_procedure(j_file,
                             TARGET='cifar10',
                             NUM_CLASSES=10):
    if TARGET == 'cifar100':
        dataloader, _ = get_cifar100_dataloaders(batch_size=128, num_workers=0)
    elif TARGET == 'ImageNet16-120':
        dataloader, _ = get_imagenet16_dataloaders(batch_size=64,
                                                   num_workers=0)
    elif TARGET == 'cifar10':
        dataloader, _ = get_cifar10_dataloaders(batch_size=64, num_workers=0)

    # grad-cam not cam
    tnet = get_nb101_teacher()

    dataiter = iter(dataloader)
    img, label = next(dataiter)
    criterion_ce = nn.CrossEntropyLoss()
    # criterion_kl = nn.KLDivLoss()
    criterion_ickd = ICKDLoss()
    criterion_sp = Similarity()

    gt_list = []
    zcs_list = []

    for k, v in j_file.items():
        # get random subnet
        snet, acc = get_nb101_model_and_acc(k)
        network_weight_gaussian_init(snet)
        network_weight_gaussian_init(tnet)
        gt_list.append(float(v))

        try:
            img, label = next(dataiter)
        except StopIteration as e:
            dataiter = iter(dataloader)
            img, label = next(dataiter)

        tfeature, tlogits = tnet.forward_with_features(img)
        sfeature, slogits = snet.forward_with_features(img)

        criterion_ce(tlogits, label).backward()
        criterion_ce(slogits, label).backward()
        tcompressed = tnet.classifier.weight.grad.unsqueeze(-1).unsqueeze(-1)
        scompressed = snet.classifier.weight.grad.unsqueeze(-1).unsqueeze(-1)

        # score_sp = -1 * criterion_sp(tfeature[-1],
        #                              sfeature[-1])[0].detach().numpy()

        score_ickd = -1 * criterion_ickd([tcompressed],
                                         [scompressed])[0].detach().numpy()
        result = score_ickd
        zcs_list.append(result if isinstance(result, float) else result[0])

    # def min_max_scale(x):
    #     return (x - np.min(x)) / (np.max(x) - np.min(x))
    
    # zcs_list = min_max_scale(zcs_list)
    
    print(
        f'kd: {kendalltau(zcs_list,gt_list):.4f} sp: {spearman(zcs_list,gt_list):.4f} ps: {pearson(zcs_list, gt_list):.4f}'
    )

    return spearman(zcs_list, gt_list)


def compute_vanilla_zc_rank(dataloader,
                            zc_name,
                            j_file=None):
    gt_list = []
    zcs_list = []

    # k is hash, v is dacc 
    for k, v in j_file.items():
        # get random subnet
        snet, acc = get_nb101_model_and_acc(k)
        network_weight_gaussian_init(snet)

        dataload_info = ['random', 3, NUM_CLASSES]
        gt_list.append(float(v))

        score = predictive.find_measures(snet,
                                         dataloader,
                                         dataload_info=dataload_info,
                                         measure_names=[zc_name],
                                         loss_fn=F.cross_entropy,
                                         device=torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda'))

        zcs_list.append(score)

    print(
        f'{zc_name} kd: {kendalltau(zcs_list,gt_list):.4f} sp: {spearman(zcs_list,gt_list):.4f} ps: {pearson(zcs_list, gt_list):.4f}'
    )

    return kendalltau(zcs_list,
                      gt_list), spearman(zcs_list,
                                         gt_list), pearson(zcs_list, gt_list)

def main_vanilla_zc(TARGET, j_file):
    if TARGET == 'cifar100':
        train_loader, val_loader = get_cifar100_dataloaders(batch_size=128,
                                                            num_workers=2)
    elif TARGET == 'ImageNet16-120':
        train_loader, val_loader = get_imagenet16_dataloaders(batch_size=64,
                                                              num_workers=2)
    elif TARGET == 'cifar10':
        train_loader, _ = get_cifar10_dataloaders(batch_size=64, num_workers=2)

    # COMPUTE NORMAL KD RANK CONSISTENCY
    times = 6

    # zc_names = ['epe_nas', 'fisher', 'grad_norm', 'grasp', 'jacov', 'l2_norm',
    #             'nwot', 'plain', 'snip', 'synflow', 'flops', 'params']

    zc_names = ['flops', 'fisher', 'grad_norm', 'snip', 'synflow', 'nwot']

    for zc_name in zc_names:
        every_time_sp = []
        every_time_kd = []
        every_time_ps = []
        for _ in range(times):
            kd, sp, ps = compute_vanilla_zc_rank(
                train_loader, zc_name=zc_name, j_file=j_file)
            every_time_sp.append(sp)
            every_time_kd.append(kd)
            every_time_ps.append(ps)
        print(f'NAME: {zc_name} LIST: {every_time_sp}')



def main_kd_zc(TARGET, NUM_CLASSES, j_file):
    if TARGET == 'cifar100':
        train_loader, val_loader = get_cifar100_dataloaders(batch_size=128,
                                                            num_workers=0)
    elif TARGET == 'ImageNet16-120':
        train_loader, val_loader = get_imagenet16_dataloaders(batch_size=64,
                                                              num_workers=0)
    elif TARGET == 'cifar10':
        train_loader, _ = get_cifar10_dataloaders(batch_size=64, num_workers=0)
    # COMPUTE NORMAL KD RANK CONSISTENCY
    times = 6

    loss_dict = {}
    # 'KD', 'FitNet', 'RKD', 'PKT', 'CC', 'AT', 'NST', 
    loss_name_list = ['NST']
    # loss_name_list = ['CC', 'KD', 'NST']
    for loss_name in loss_name_list:
        every_time_sp = []
        every_time_kd = []
        every_time_ps = []
        for _ in range(times):
            # try:
            kd, sp, ps = compute_vanilla_distiller_rank(
                train_loader, loss_name, j_file, TARGET, NUM_CLASSES)
            every_time_sp.append(sp)
            every_time_kd.append(kd)
            every_time_ps.append(ps)

        print(f'LOSS: {loss_name} LIST: {every_time_sp}')
        loss_dict[loss_name] = {
            'kd': every_time_kd,
            'sp': every_time_sp,
            'ps': every_time_ps,
        }

    with open(f'./kd_loss_name-result_x10_{TARGET}.txt', 'w') as f:
        f.write(json.dumps(loss_dict))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--target',
                        type=str,
                        default='cifar100',
                        help='print frequency')
    opt = parser.parse_args()

    TARGET = opt.target

    if TARGET == 'cifar100':
        NUM_CLASSES = 100
    elif TARGET == 'ImageNet16-120':
        NUM_CLASSES = 120
    elif TARGET == 'cifar10':
        NUM_CLASSES = 10

    # COMPUTE NORMAL KD RANK CONSISTENCY
    print(f'TARGET: {TARGET} / NUM_CLASSES: {NUM_CLASSES}')

    with open('./data/dist_nb101_bench.json', 'r') as f:
        j_file = json.load(f)

    # t1 = time.time()
    # print("=====> diswot under distill 101 benchmark")
    # times = 5
    # res = []
    # for i in range(times):
    #     sp = compute_diswot_procedure(j_file)
    #     res.append(sp)
    # print(res)
    # print(f'=====> time: {(time.time() - t1)/5} s')

    # print("=====> zc under distill 101 benchmark")
    # main_vanilla_zc(TARGET, j_file)

    print("=====> KDzc under distill 101 benchmark")
    main_kd_zc(TARGET, NUM_CLASSES, j_file)