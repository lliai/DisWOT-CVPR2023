import os
import random
import sys
import time
import argparse 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from dataset.cifar100 import get_cifar100_dataloaders
from distiller_zoo import ICKDLoss, RMIloss, Similarity
from models.candidates.fixed_models.resnet import resnet56
from models.nasbench201.utils import dict2config, get_cell_based_tiny_net
from predictor.pruners import predictive
import torch.nn.functional as F 

from models.nasbench101.build import (S50_v0, S50_v1, S50_v2,
                                      get_nb101_model_and_acc,
                                      get_nb101_teacher, get_rnd_nb101_and_acc, query_nb101_acc)

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

def search_best_subnet_by_diswot(dataloader, iterations=None):
    # tnet = get_teacher_best_model()
    tnet = get_nb101_teacher()
    network_weight_gaussian_init(tnet)

    dataiter = iter(dataloader)
    img, label = next(dataiter)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_ickd = ICKDLoss()
    criterion_sp = Similarity()

    sample_number = 1000 if iterations is None else iterations
    best_score = -1e9
    best_arch_hash = None

    for idx in range(sample_number):
        snet, acc, arch_hash = get_rnd_nb101_and_acc()
        network_weight_gaussian_init(snet)

        tfeature, tlogits = tnet(img, is_feat=True)
        sfeature, slogits = snet.forward_with_features(img)

        criterion_ce(tlogits, label).backward()
        criterion_ce(slogits, label).backward()

        tcompressed = tnet.classifier.weight.unsqueeze(-1).unsqueeze(-1)
        scompressed = snet.classifier.weight.unsqueeze(-1).unsqueeze(-1)

        # score1 = -1 * \
        #     criterion_sp(tfeature[-2], sfeature[-2])[0].detach().numpy()
        score2 = -1 * criterion_ickd([tcompressed],
                                     [scompressed])[0].detach().numpy()

        score = score2
        if score > best_score:
            best_score = score
            best_arch_hash = arch_hash
        
        print(f'iter: {idx} current arch: {arch_hash} current acc: {query_nb101_acc(arch_hash)}')

    print(
        f'Best arch found by our metric: {best_arch_hash} gt acc: {query_nb101_acc(best_arch_hash)}'
    )

def search_best_subnet_by_vanillazc(dataloader, iterations, zc_name):
    dataiter = iter(dataloader)
    img, label = next(dataiter)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_ickd = ICKDLoss()
    criterion_sp = Similarity()

    sample_number = 1000 if iterations is None else iterations
    best_score = -1e9
    best_arch_hash = None

    for idx in range(sample_number):
        snet, acc, arch_hash = get_rnd_nb101_and_acc()
        network_weight_gaussian_init(snet)

        if zc_name is not 'grasp':
            dataload_info = ['random', 3, 100]
        else:
            dataload_info = ['grasp', 3, 100]

        score = predictive.find_measures(snet,
                                         dataloader,
                                         dataload_info=dataload_info,
                                         measure_names=[zc_name],
                                         loss_fn=F.cross_entropy,
                                         device=torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda'))
        if score > best_score:
            best_score = score
            best_arch_hash = arch_hash
        
        if idx % 10 == 0:
            print(f'iter: {idx} current arch: {arch_hash} current acc: {query_nb101_acc(arch_hash)}')

    print(
        f'Best arch found by our metric: {best_arch_hash} gt acc: {query_nb101_acc(best_arch_hash)}'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Random Search Algorithm')
    parser.add_argument('--iterations',
                        type=int, 
                        default=1000, 
                        help='iterations remains to search')

    parser.add_argument('--zc_name',
                        type=str,
                        default='zen',
                        help='zero cost proxy name')

    args = parser.parse_args()

    train_loader, val_loader = get_cifar100_dataloaders(batch_size=64,
                                                        num_workers=2)

    t1 = time.time()
    # search_best_subnet_by_diswot(train_loader, args.iterations)
    # search_best_subnet_by_vanillazc(train_loader, args.iterations, args.zc_name)

    zc_name_list = ['snip', 'grasp', 'nwot', 'synflow']

    for zc_name in zc_name_list:
        print(f'==== PROCESSING {zc_name} ====')
        search_best_subnet_by_vanillazc(train_loader, args.iterations, zc_name)

    print(f'==== PROCESSING DISWOT ====')
    search_best_subnet_by_diswot(train_loader, args.iterations)


    print(f'Search time: {time.time() - t1} s')
