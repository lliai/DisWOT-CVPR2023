import os
import random
import sys
import time
from copy import deepcopy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import collections

import torch
from nas_201_api import NASBench201API
from nas_201_api import NASBench201API as API
from utils import (AverageMeter, load_config, obtain_accuracy, prepare_logger,
                   prepare_seed, time_string)

from dataset.nb201_data import get_datasets
from distiller_zoo import ICKDLoss, Similarity
from helper.utils.flop_benchmark import get_model_infos
from models.candidates.fixed_models.resnet import resnet56
from models.nasbench201.utils import (CellStructure, dict2config,
                                      get_cell_based_tiny_net)
from predictor.pruners import predictive


nb201_api = NASBench201API(
    file_path_or_dict='data/NAS-Bench-201-v1_0-e61699.pth', verbose=False)


class Model(object):
    def __init__(self):
        self.arch = None
        self.accuracy = None

    def __str__(self):
        """Prints a readable version of this bitstring."""
        return '{:}'.format(self.arch)

    def __repr__(self):
        return f'Model(arch={self.arch} acc={self.accuracy})'


def valid_func(xloader, network, criterion):
    data_time, batch_time = AverageMeter(), AverageMeter()
    arch_losses, arch_top1, arch_top5 = AverageMeter(), AverageMeter(
    ), AverageMeter()
    network.train()
    end = time.time()
    with torch.no_grad():
        for step, (arch_inputs, arch_targets) in enumerate(xloader):
            arch_targets = arch_targets.cuda(non_blocking=True)
            # measure data loading time
            data_time.update(time.time() - end)
            # prediction
            _, logits = network(arch_inputs)
            arch_loss = criterion(logits, arch_targets)
            # record
            arch_prec1, arch_prec5 = obtain_accuracy(logits.data,
                                                     arch_targets.data,
                                                     topk=(1, 5))
            arch_losses.update(arch_loss.item(), arch_inputs.size(0))
            arch_top1.update(arch_prec1.item(), arch_inputs.size(0))
            arch_top5.update(arch_prec5.item(), arch_inputs.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    return arch_losses.avg, arch_top1.avg, arch_top5.avg


def get_model_by_arch(arch):
    arch_config = {
        'name': 'infer.tiny',
        'C': 16,
        'N': 5,
        'arch_str': arch,
        'num_classes': 100
    }
    net_config = dict2config(arch_config, None)
    return get_cell_based_tiny_net(net_config)


def train_and_eval(arch,
                   nas_bench=None,
                   zc_proxy: str = 'nwot',
                   train_loader=None):
    """_summary_

    Args:
        arch (_type_): _description_
        nas_bench (_type_, optional): _description_. Defaults to None.
        zerocost (str, optional): _description_. Defaults to 'nwot'.

    Available zc:
        'epe_nas', 'fisher', 'grad_norm', 'grasp', 'jacov', 'l2_norm',
        'nwot', 'plain', 'snip', 'synflow', 'flops', 'params', 'diswot', 'zen'
    """
    # print(f'eval {arch} with zc_proxy {zc_proxy}')
    if nas_bench is not None:
        arch_index = nas_bench.query_index_by_arch(arch)
        assert arch_index >= 0, 'can not find this arch : {:}'.format(arch)
        info = nas_bench.get_more_info(arch_index, 'cifar10-valid', None,
                                       '200')
        reward, time_cost = info['valid-accuracy'], 1
        FLOPs, Param = -1, -1
        # not supported
    else:
        assert zc_proxy is not None
        if zc_proxy == 'diswot':
            tnet = resnet56(num_classes=10)

            import torch.nn as nn
            criterion_ce = nn.CrossEntropyLoss()
            criterion_sp = Similarity()
            criterion_ickd = ICKDLoss()

            dataiter = iter(train_loader)
            img, label = next(dataiter)

            snet = get_model_by_arch(arch.tostr())

            tfeature, tlogits = tnet(img, is_feat=True)
            # tnet.forward_with_features(img)
            sfeature, slogits = snet.forward_with_features(img)

            criterion_ce(tlogits, label).backward()
            criterion_ce(slogits, label).backward()

            tcompressed = tnet.fc.weight.grad.unsqueeze(-1).unsqueeze(-1)
            scompressed = snet.classifier.weight.grad.unsqueeze(-1).unsqueeze(
                -1)

            # score1 = -1 * criterion_sp(tfeature[-2],
            #                            sfeature[-2])[0].detach().numpy()
            score2 = -1 * criterion_ickd([tcompressed],
                                         [scompressed])[0].detach().numpy()

            FLOPs, Param = get_model_infos(snet, shape=(1, 3, 32, 32))
            reward = score2  #  + score2
            time_cost = 1
        else:
            import torch.nn.functional as F

            snet = get_model_by_arch(arch.tostr())
            if isinstance(zc_proxy, str):
                zc_proxy = [zc_proxy]
            reward = predictive.find_measures(snet,
                                              train_loader,
                                              dataload_info=['random', 3, 100],
                                              measure_names=zc_proxy,
                                              loss_fn=F.cross_entropy,
                                              device=torch.device('cuda:0'))
            FLOPs, Param = get_model_infos(snet, shape=(1, 3, 32, 32))
            print(FLOPs)
            time_cost = 1

    return reward, time_cost, FLOPs, Param


def random_architecture_func(max_nodes, op_names):
    # return a random architecture
    def random_architecture():
        genotypes = []
        for i in range(1, max_nodes):
            xlist = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                op_name = random.choice(op_names)
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return CellStructure(genotypes)

    return random_architecture


def mutate_arch_func(op_names):
    """Computes the architecture for a child of the given parent
     architecture. The parent architecture is cloned and mutated 
     to produce the child architecture. The child architecture is
     mutated by randomly switch one operation to another.
  """
    def mutate_arch_func(parent_arch):
        child_arch = deepcopy(parent_arch)
        node_id = random.randint(0, len(child_arch.nodes) - 1)
        node_info = list(child_arch.nodes[node_id])
        snode_id = random.randint(0, len(node_info) - 1)
        xop = random.choice(op_names)
        while xop == node_info[snode_id][0]:
            xop = random.choice(op_names)
        node_info[snode_id] = (xop, node_info[snode_id][1])
        child_arch.nodes[node_id] = tuple(node_info)
        return child_arch

    return mutate_arch_func


def get_nb201_gt(arch_str):
    choiced_index = nb201_api.query_index_by_arch(arch_str)
    xinfo = nb201_api.get_more_info(choiced_index,
                                    dataset='cifar100',
                                    hp='200')
    return xinfo['test-accuracy']


def regularized_evolution(cycles, population_size, sample_size, time_budget,
                          random_arch, mutate_arch, nas_bench, extra_info,
                          train_loader, zc_proxy, logger, TARGET_FLOPS=None):
    """Algorithm for regularized evolution (i.e. aging evolution).
    Follows "Algorithm 1" in Real et al. "Regularized Evolution for Image
    Classifier Architecture Search".

    Args:
        cycles: the number of cycles the algorithm should run for.
        population_size: the number of individuals to keep in the population.
        sample_size: the number of individuals that should participate in each tournament.
        time_budget: the upper bound of searching cost

    Returns:
        history: a list of `Model` instances, representing all the models computed
            during the evolution experiment.
  """
    population = collections.deque()
    # Not used by the algorithm, only used to report results.
    history, total_time_cost = [], 0
    MODEL_FLOPS = 1e10
    TARGET_FLOPS = 50 if TARGET_FLOPS is None else TARGET_FLOPS
    # 0.4

    # Initialize the population with random models.
    while len(population) < population_size:
        while MODEL_FLOPS > TARGET_FLOPS:
            model = Model()
            model.arch = random_arch()
            model.accuracy, time_cost, MODEL_FLOPS, params = train_and_eval(
                model.arch,
                nas_bench,
                zc_proxy=zc_proxy,
                train_loader=train_loader)
            print("current flops is: ====> ", MODEL_FLOPS)

        population.append(model)
        history.append(model)
        total_time_cost += time_cost
        MODEL_FLOPS = -1

    # Carry out evolution in cycles. Each cycle produces a model and removes
    # another.
    # while len(history) < cycles:
    while total_time_cost < time_budget:
        print(f'current tried times: {len(history)}/{time_budget}')
        # Sample randomly chosen models from the current population.
        start_time, sample = time.time(), []
        while len(sample) < sample_size:
            # Inefficient, but written this way for clarity. In the case of neural
            # nets, the efficiency of this line is irrelevant because training neural
            # nets is the rate-determining step.
            candidate = random.choice(list(population))
            sample.append(candidate)

        # The parent is the best model in the sample.
        sample = sorted(sample, key=lambda i: i.accuracy)
        top_N = 5
        parent = random.choice(
            sample[:top_N])  # max(sample, key=lambda i: i.accuracy)

        # Create the child model and store it.
        child = Model()
        child.arch = mutate_arch(parent.arch)
        total_time_cost += time.time() - start_time
        child.accuracy, time_cost, flops, params = train_and_eval(
            child.arch,
            nas_bench,
            zc_proxy=zc_proxy,
            train_loader=train_loader)
        if total_time_cost + time_cost > time_budget:  # return
            return history, total_time_cost
        else:
            total_time_cost += time_cost
        population.append(child)
        history.append(child)

        # Remove the oldest model.
        population.popleft()
    return history, total_time_cost


def main(xargs, nas_bench):
    assert torch.cuda.is_available(), 'CUDA is not available.'
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(xargs.workers)
    prepare_seed(xargs.rand_seed)
    logger = prepare_logger(args)

    assert xargs.dataset == 'cifar10', \
        f'currently only support CIFAR-10 but got {xargs.dataset}'
    if xargs.data_path is not None:
        train_data, valid_data, xshape, class_num = get_datasets(
            xargs.dataset, xargs.data_path, -1)

        split_Fpath = 'configs/cifar-split.txt'
        cifar_split = load_config(split_Fpath, None, None)
        train_split, valid_split = cifar_split.train, cifar_split.valid
        print('Load split file from {:}'.format(split_Fpath))
        config_path = 'configs/R-EA.config'
        config = load_config(config_path, {
            'class_num': class_num,
            'xshape': xshape
        }, logger)
        # To split data
        train_data_v2 = deepcopy(train_data)
        train_data_v2.transform = valid_data.transform
        valid_data = train_data_v2

        # data loader
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=config.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(train_split),
            num_workers=xargs.workers,
            pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(
            valid_data,
            batch_size=config.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(valid_split),
            num_workers=xargs.workers,
            pin_memory=True)
        print(
            '||||||| {:10s} ||||||| Train-Loader-Num={:}, Valid-Loader-Num={:}, batch size={:}'
            .format(xargs.dataset, len(train_loader), len(valid_loader),
                    config.batch_size))
        print('||||||| {:10s} ||||||| Config={:}'.format(
            xargs.dataset, config))
        extra_info = {
            'config': config,
            'train_loader': train_loader,
            'valid_loader': valid_loader
        }
    else:
        config_path = 'configs/R-EA.config'
        config = load_config(config_path, None, logger)
        print('||||||| {:10s} ||||||| Config={:}'.format(
            xargs.dataset, config))
        extra_info = {
            'config': config,
            'train_loader': None,
            'valid_loader': None
        }

    # search_space = get_search_spaces('cell', xargs.search_space_name)
    search_space = [
        'none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3'
    ]
    random_arch = random_architecture_func(xargs.max_nodes, search_space)
    mutate_arch = mutate_arch_func(search_space)

    x_start_time = time.time()
    print('{:} use nas_bench : {:}'.format(time_string(), nas_bench))
    print('-' * 30 +
               ' start searching with the time budget of {:} s'.format(
                   xargs.time_budget))

    history, total_cost = regularized_evolution(
        xargs.ea_cycles, xargs.ea_population, xargs.ea_sample_size,
        xargs.time_budget, random_arch, mutate_arch,
        nas_bench if args.ea_fast_by_api else None, extra_info, train_loader,
        args.zc_proxy, logger, TARGET_FLOPS=xargs.target_flops)

    x = []
    y = []
    for i, m in enumerate(history):
        x.append(i)
        y.append(m.accuracy)

    import matplotlib.pyplot as plt
    plt.plot(x, y)
    plt.savefig(f'evo_nb201_results_{xargs.zc_proxy}.png')

    print(
        '{:} regularized_evolution finish with history of {:} arch with {:.1f} s (real-cost={:.2f} s).'
        .format(time_string(), len(history), total_cost,
                time.time() - x_start_time))
    top_N = 5
    history = sorted(history, key=lambda i: i.accuracy, reverse=True)
    best_acc = -1
    best_arch = None

    for h in history[:top_N]:
        current_acc = get_nb201_gt(h.arch)
        if best_acc < current_acc:
            best_acc = current_acc
            best_arch = h.arch

    # best_arch = max(history, key=lambda i: i.accuracy)
    # best_arch = best_arch.arch
    print('{:} best arch is {:} acc is {:}'.format(time_string(), best_arch, best_acc))

    info = nas_bench.query_by_arch(best_arch, hp='200')
    if info is None:
        print('Did not find this architecture : {:}.'.format(best_arch))
    else:
        print('{:}'.format(info))
    print('-' * 100)
    logger.close()
    return nas_bench.query_index_by_arch(best_arch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Regularized Evolution Algorithm')
    parser.add_argument('--data_path',
                        type=str,
                        default='./data',
                        help='Path to dataset')
    parser.add_argument('--dataset',
                        type=str,
                        default='cifar10',
                        choices=['cifar10', 'cifar100', 'ImageNet16-120'],
                        help='Choose between Cifar10/100 and ImageNet-16.')
    parser.add_argument('--max_nodes',
                        type=int,
                        default=4,
                        help='The maximum number of nodes.')
    parser.add_argument('--channel',
                        type=int,
                        default=16,
                        help='The number of channels.')
    parser.add_argument('--num_cells',
                        type=int,
                        default=5,
                        help='The number of cells in one stage.')
    parser.add_argument('--ea_cycles',
                        type=int,
                        default=1000,
                        help='The number of cycles in EA.')
    parser.add_argument('--ea_population',
                        type=int,
                        default=36,
                        help='The population size in EA.')
    parser.add_argument('--ea_sample_size',
                        type=int,
                        default=8,
                        help='The sample size in EA.')
    parser.add_argument('--ea_fast_by_api',
                        type=int,
                        default=0,
                        help='Use our API to speed up the experiments or not.')
    parser.add_argument(
        '--time_budget',
        type=int,
        default=1000,
        help='The total time cost budge for searching (in seconds).')
    # log
    parser.add_argument('--workers',
                        type=int,
                        default=2,
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--save_dir',
                        type=str,
                        default='./save_dirs/evo_nb201',
                        help='Folder to save checkpoints and log.')
    parser.add_argument(
        '--arch_nas_dataset',
        type=str,
        default='./data/NAS-Bench-201-v1_0-e61699.pth',
        help='The path to load the architecture dataset (tiny-nas-benchmark).')
    parser.add_argument('--print_freq',
                        type=int,
                        help='print frequency (default: 200)')
    parser.add_argument('--rand_seed',
                        type=int,
                        default=42,
                        help='manual seed')
    parser.add_argument('--zc_proxy',
                        type=str,
                        default='diswot',
                        help='zero cost proxy name')
    
    parser.add_argument('--target_flops',
                        type=float, 
                        default=100, 
                        help='search models under target flops')

    args = parser.parse_args()
    args.ea_fast_by_api = args.ea_fast_by_api > 0

    if args.arch_nas_dataset is None or not os.path.isfile(
            args.arch_nas_dataset):
        nas_bench = None
    else:
        print('{:} build NAS-Benchmark-API from {:}'.format(
            time_string(), args.arch_nas_dataset))
        nas_bench = API(args.arch_nas_dataset)

    if args.rand_seed < 0:
        save_dir, all_indexes, num = None, [], 500
        for i in range(num):
            print('{:} : {:03d}/{:03d}'.format(time_string(), i, num))
            args.rand_seed = random.randint(1, 100000)
            save_dir, index = main(args, nas_bench)
            all_indexes.append(index)
        torch.save(all_indexes, save_dir / 'results.pth')
    else:
        main(args, nas_bench)
