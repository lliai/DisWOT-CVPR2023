import os
import random
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import torch.nn as nn
from nas_201_api import NASBench201API

from dataset.cifar100 import get_cifar100_dataloaders
from distiller_zoo import ICKDLoss, RMIloss, Similarity
from models.candidates.fixed_models.resnet import resnet56
from models.nasbench201.utils import dict2config, get_cell_based_tiny_net
from predictor.pruners import predictive
import torch.nn.functional as F 
from helper.utils.flop_benchmark import get_model_infos

# from rank_consisteny import kendalltau, pearson, spearman

nb201_api = NASBench201API(
    file_path_or_dict='data/NAS-Bench-201-v1_0-e61699.pth', verbose=False)


def get_teacher_best_model():
    best_idx, high_accurcy = nb201_api.find_best(dataset='cifar100',
                                                 metric_on_set='test',
                                                 hp='200')
    arch_config = {
        'name': 'infer.tiny',
        'C': 16,
        'N': 5,
        'arch_str': nb201_api.arch(best_idx),
        'num_classes': 100
    }
    net_config = dict2config(arch_config, None)
    print(
        f'find teacher idx: {best_idx}; arch_str: {nb201_api.arch(best_idx)}')
    return get_cell_based_tiny_net(net_config)


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


def random_sample_and_get_gt():
    index_range = list(range(15625))
    choiced_index = random.choice(index_range)
    # modelinfo is a index
    # assert choiced_index is not None
    # modelinfo = 15624
    arch_config = {
        'name': 'infer.tiny',
        'C': 16,
        'N': 5,
        'arch_str': nb201_api.arch(choiced_index),
        'num_classes': 100
    }
    net_config = dict2config(arch_config, None)
    model = get_cell_based_tiny_net(net_config)
    xinfo = nb201_api.get_more_info(choiced_index,
                                    dataset='cifar100',
                                    hp='200')
    FLOPs, Param = get_model_infos(model, shape=(1, 3, 32, 32))

    return choiced_index, model, xinfo['test-accuracy'], FLOPs, Param


def search_best_subnet(dataloader, iterations=None):
    # tnet = get_teacher_best_model()
    tnet = resnet56(num_classes=100)
    network_weight_gaussian_init(tnet)

    dataiter = iter(dataloader)
    img, label = next(dataiter)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_ickd = ICKDLoss()
    criterion_sp = Similarity()

    sample_number = 1000 if iterations is None else iterations
    best_score = -1e9
    best_arch_index = -1
    seen_list = []

    for _ in range(sample_number):
        idx, snet, acc, flops, params = random_sample_and_get_gt()
        # find snet under 50M FLOPs
        while flops > 50:
            idx, snet, acc, flops, params = random_sample_and_get_gt()
        network_weight_gaussian_init(snet)

        tfeature, tlogits = tnet(img, is_feat=True)
        # tnet.forward_with_features(img)
        sfeature, slogits = snet.forward_with_features(img)

        criterion_ce(tlogits, label).backward()
        criterion_ce(slogits, label).backward()

        tcompressed = tnet.fc.weight.unsqueeze(-1).unsqueeze(-1)
        scompressed = snet.classifier.weight.unsqueeze(-1).unsqueeze(-1)

        # score1 = -1 * \
        #     criterion_sp(tfeature[-2], sfeature[-2])[0].detach().numpy()
        score2 = -1 * criterion_ickd([tcompressed],
                                     [scompressed])[0].detach().numpy()

        score = score2
        if score > best_score:
            best_score = score
            best_arch_index = idx

    xinfo = nb201_api.get_more_info(best_arch_index,
                                    dataset='cifar100',
                                    hp='200')

    print(
        f'Best index found by our metric: {best_arch_index} arch str: {nb201_api.arch(best_arch_index)} info: {xinfo}'
    )

def search_best_subnet_by_vanillazc(dataloader, iterations, zc_name):
    dataiter = iter(dataloader)
    img, label = next(dataiter)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_ickd = ICKDLoss()
    criterion_sp = Similarity()

    sample_number = iterations
    best_score = -1e9
    best_arch_index = -1

    for _ in range(sample_number):
        idx, snet, acc, flops, params = random_sample_and_get_gt()
        # find snet under 50M FLOPs
        while flops > 50:
            idx, snet, acc, flops, params = random_sample_and_get_gt()
        network_weight_gaussian_init(snet)

        if zc_name != 'grasp':
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
            best_arch_index = idx

    xinfo = nb201_api.get_more_info(best_arch_index,
                                    dataset='cifar100',
                                    hp='200')

    print(
        f'Best index found by our metric: {best_arch_index} arch str: {nb201_api.arch(best_arch_index)} info: {xinfo}'
    )


def search_best_subnet_faster(dataloader):
    tnet = get_teacher_best_model()
    # tnet = resnet56(num_classes=100)
    network_weight_gaussian_init(tnet)

    dataiter = iter(dataloader)
    img, label = next(dataiter)
    criterion_sp = Similarity()

    sample_number = 5000
    best_score = -1e9
    best_arch_index = -1
    seen_list = []

    for _ in range(sample_number):
        idx, snet, acc = random_sample_and_get_gt()
        if idx in seen_list:
            idx, snet, acc = random_sample_and_get_gt()
        else:
            seen_list.append(idx)
        network_weight_gaussian_init(snet)

        tfeature, tlogits = tnet(img, is_feat=True)
        sfeature, slogits = snet.forward_with_features(img)

        score = -1 * \
            criterion_sp(tfeature[-2], sfeature[-2])[0].detach().numpy()

        if score > best_score:
            best_score = score
            best_arch_index = idx

    xinfo = nb201_api.get_more_info(best_arch_index,
                                    dataset='cifar100',
                                    hp='200')

    print(
        f'Best index found by our metric: {best_arch_index} arch str: {nb201_api.arch(best_arch_index)} xinfo: {xinfo}'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Random Search Algorithm')
    parser.add_argument('--iterations',
                        type=int, 
                        default=100, 
                        help='iterations remains to search')

    parser.add_argument('--zc_name',
                        type=str,
                        default='zen',
                        help='zero cost proxy name')

    args = parser.parse_args()
    train_loader, val_loader = get_cifar100_dataloaders(batch_size=64,
                                                        num_workers=2)

    t1 = time.time()
    # search_best_subnet(train_loader)
    # search_best_subnet_faster(train_loader)

    zc_names = ['snip', 'grasp', 'nwot', 'synflow']
    for zc_name in zc_names:
        print(f'==== PROCESSING {zc_name} ====')
        search_best_subnet_by_vanillazc(train_loader, args.iterations, zc_name)

    print(f'==== PROCESSING DISWOT ====')
    search_best_subnet(train_loader, args.iterations)
    print(f'Search time: {time.time() - t1} s')
