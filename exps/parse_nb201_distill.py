import argparse
import os
import random
import sys
import time 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from typing import List

import matplotlib.pyplot as plt
import seaborn
import torch
import torch.nn as nn
import torch.nn.functional as F
from nas_201_api import NASBench201API
from rank_consisteny import kendalltau, pearson, spearman
from tqdm import tqdm

from dataset.cifar10 import (get_cifar10_dataloaders,
                             get_cifar10_dataloaders_entropy)
from dataset.cifar100 import (get_cifar100_dataloaders,
                              get_cifar100_dataloaders_entropy)
from dataset.imagenet16 import get_imagenet16_dataloaders
from distiller_zoo import (Attention, Correlation, ICKDLoss, RKDLoss, RMIloss,
                           Similarity, build_loss)
from models import ResNet50, resnet56, resnet110
from models.nasbench201.utils import dict2config, get_cell_based_tiny_net
from predictor.pruners import predictive

nb201_api = NASBench201API(
    file_path_or_dict='data/NAS-Bench-201-v1_0-e61699.pth', verbose=False)

def get_teacher_best_model(TARGET, NUM_CLASSES):
    best_idx, high_accurcy = nb201_api.find_best(
        dataset=TARGET,  # ImageNet16-120
        metric_on_set='test',
        hp='200')
    arch_config = {
        'name': 'infer.tiny',
        'C': 16,
        'N': 5,
        'arch_str': nb201_api.arch(best_idx),
        'num_classes': NUM_CLASSES
    }
    net_config = dict2config(arch_config, None)
    return get_cell_based_tiny_net(net_config)


def random_sample_and_get_gt():
    index_range = list(range(15625))
    choiced_index = random.choice(index_range)
    # modelinfo is a index
    # modelinfo = 15624
    arch_config = {
        'name': 'infer.tiny',
        'C': 16,
        'N': 5,
        'arch_str': nb201_api.arch(choiced_index),
        'num_classes': NUM_CLASSES
    }
    net_config = dict2config(arch_config, None)
    model = get_cell_based_tiny_net(net_config)
    xinfo = nb201_api.get_more_info(choiced_index, dataset=TARGET, hp='200')
    return choiced_index, model, xinfo['test-accuracy']


def get_network_by_index(choiced_index):
    arch_config = {
        'name': 'infer.tiny',
        'C': 16,
        'N': 5,
        'arch_str': nb201_api.arch(choiced_index),
        'num_classes': 100
    }
    net_config = dict2config(arch_config, None)
    model = get_cell_based_tiny_net(net_config)
    xinfo = nb201_api.get_more_info(choiced_index, dataset='cifar100', hp='200')
    return model, xinfo['test-accuracy']

def visualize_figures(gt_list: List, set_list: List[list], title: List = None):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(20, 4))

    for i in range(5):
        ax = fig.add_subplot(15 * 10 + i + 1)
        ax.scatter(gt_list, set_list[i])
        ax.set_title(f'{title[i]}_kd:{kendalltau(gt_list, set_list[i]):.2f}')
    plt.savefig('./results_kd.png')


def visulize_sp_G(f_s, f_t, name: str = None):
    bsz = f_s.shape[0]
    f_s = f_s.view(bsz, -1)
    f_t = f_t.view(bsz, -1)

    G_s = torch.mm(f_s, torch.t(f_s))
    # G_s = G_s / G_s.norm(2)
    G_s = torch.nn.functional.normalize(G_s)
    G_t = torch.mm(f_t, torch.t(f_t))
    # G_t = G_t / G_t.norm(2)
    G_t = torch.nn.functional.normalize(G_t)

    G_diff = G_t - G_s

    # normalize to 0-1
    min_a = torch.min(G_diff)
    max_a = torch.max(G_diff)
    G_diff = (G_diff - min_a) / (max_a - min_a)
    seaborn.heatmap(G_diff, vmin=0, vmax=1.0, cmap='coolwarm')  # summer

    plt.axis('off')
    plt.savefig(name, dpi=200)
    plt.clf()


def visulize_ickd_G(f_s, f_t, name: str = None):
    bsz, ch = f_s.shape[0], f_s.shape[1]

    f_s = f_s.view(bsz, ch, -1)
    f_t = f_t.view(bsz, ch, -1)

    emd_s = torch.bmm(f_s, f_s.permute(0, 2, 1))
    emd_s = torch.nn.functional.normalize(emd_s, dim=2)

    emd_t = torch.bmm(f_t, f_t.permute(0, 2, 1))
    emd_t = torch.nn.functional.normalize(emd_t, dim=2)

    G_diff = emd_s - emd_t

    # normalize to 0-1
    G_diff = torch.mean(G_diff, dim=0)
    min_a = torch.min(G_diff)
    max_a = torch.max(G_diff)
    G_diff = (G_diff - min_a) / (max_a - min_a)

    seaborn.heatmap(G_diff, vmin=0, vmax=1.0, cmap='summer')  # summer
    plt.axis('off')
    plt.savefig(name)
    plt.clf()


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


def compute_zen_1st_replace_grad_procedure(sample_number=50,
                                           TARGET='cifar10',
                                           NUM_CLASSES=10):
    tnet = get_teacher_best_model(TARGET, NUM_CLASSES)
    network_weight_gaussian_init(tnet)

    ## device and dtype
    device = torch.device('cpu')
    dtype = torch.float32
    mixup_gamma = 1e-2

    # inputs
    input = torch.randn(size=(64, 3, 32, 32), device=device, dtype=dtype)
    input2 = torch.randn(size=(64, 3, 32, 32), device=device, dtype=dtype)
    mixup_input = input + mixup_gamma * input2

    tnet = tnet.to(device)

    # criterions
    # criterion_ce = nn.CrossEntropyLoss()
    # criterion_rmi = RMIloss(img.size()[0])
    # criterion_kl = nn.KLDivLoss()
    criterion_ickd = ICKDLoss()
    # criterion_sp = Similarity()

    # recorders
    gt_list = []
    zcs_list = []
    times = 1

    chosen_idx = []
    for i in tqdm(range(sample_number)):
        idx, snet, acc = random_sample_and_get_gt()
        snet = snet.to(device)
        gt_list.append(acc)

        tfeature1, _ = tnet.forward_with_features(input)
        tfeature2, _ = tnet.forward_with_features(mixup_input)
        tscore_tensor = torch.abs(tfeature1[-2] - tfeature2[-2])

        sfeature1, _ = snet.forward_with_features(input)
        sfeature2, _ = snet.forward_with_features(mixup_input)
        sscore_tensor = torch.abs(sfeature1[-2] - sfeature2[-2])

        tcompressed = tscore_tensor
        scompressed = sscore_tensor

        # post process.
        item1 = -1 * criterion_ickd([tcompressed],
                                    [scompressed])[0].detach().numpy()
        zcs_list.append(item1)

    print(
        f'kd: {kendalltau(zcs_list,gt_list):.4f} sp: {spearman(zcs_list,gt_list):.4f} ps: {pearson(zcs_list, gt_list):.4f}'
    )

    return f'kd: {kendalltau(zcs_list,gt_list):.4f} sp: {spearman(zcs_list,gt_list):.4f} ps: {pearson(zcs_list, gt_list):.4f}'


def compute_normal_kd_rank_procedure(dataloader,
                                     loss_name,
                                     sample_number=50,
                                     TARGET='cifar10',
                                     NUM_CLASSES=10):
    # grad-cam not cam
    tnet = get_teacher_best_model(TARGET, NUM_CLASSES)
    # network_weight_gaussian_init(tnet)

    dataiter = iter(dataloader)
    img, label = next(dataiter)
    # criterion_ce = nn.CrossEntropyLoss()

    if loss_name != 'RMI':
        criterion = build_loss(loss_name)
    else:
        criterion = RMIloss(img.size()[0])

    gt_list = []
    zcs_list = []

    chosen_idx = []
    for i in tqdm(range(sample_number)):
        # get random subnet
        idx, snet, acc = random_sample_and_get_gt()
        if idx not in chosen_idx:
            chosen_idx.append(idx)
        else:
            idx, snet, acc = random_sample_and_get_gt()
        # network_weight_gaussian_init(snet)

        gt_list.append(acc)

        try:
            img, label = next(dataiter)
        except StopIteration as e:
            dataiter = iter(dataloader)
            img, label = next(dataiter)

        tfeature, tlogits = tnet.forward_with_features(img)
        sfeature, slogits = snet.forward_with_features(img)

        # criterion_ce(tlogits, label).backward()
        # criterion_ce(slogits, label).backward()
        # tcompressed = tnet.classifier.weight.grad.unsqueeze(-1).unsqueeze(-1)
        # scompressed = snet.classifier.weight.grad.unsqueeze(-1).unsqueeze(-1)
        # visulize_ickd_G(tcompressed.detach(), scompressed.detach(), name=f'fig_of_{i}_th_subnet')

        if loss_name == 'KD':
            result = -1 * criterion(tlogits, slogits).detach().numpy()
        elif loss_name in ['FitNet', 'RKD', 'PKT', 'CC']:
            result = -1 * \
                criterion(tfeature[-1], sfeature[-1]).detach().numpy()
        elif loss_name in ['AT', 'NST', 'KDSVD']:
            result = -1 * \
                sum(criterion(tfeature[1:-1], sfeature[1:-1])).detach().numpy()
        elif loss_name == 'SP':
            result = -1 * \
                criterion(tfeature[-2], sfeature[-2])[0].detach().numpy()
        else:
            raise f'Not support {loss_name} currently.'

        zcs_list.append(result if isinstance(result, float) else result[0])

    print(
        f'{loss_name} kd: {kendalltau(zcs_list,gt_list):.4f} sp: {spearman(zcs_list,gt_list):.4f} ps: {pearson(zcs_list, gt_list):.4f}'
    )

    return kendalltau(zcs_list,
                      gt_list), spearman(zcs_list,
                                         gt_list), pearson(zcs_list, gt_list)


def visualize_diswot_sp_procedure(model1, model2, dataloader, name=None):
    network_weight_gaussian_init(model1)
    network_weight_gaussian_init(model2)
    model1.cuda()
    model2.cuda()

    dataiter = iter(dataloader)
    criterion_ce = nn.CrossEntropyLoss()

    try:
        img, label = next(dataiter)
        img, label = img.cuda(), label.cuda()
    except StopIteration as e:
        dataiter = iter(dataloader)
        img, label = next(dataiter)
        img, label = img.cuda(), label.cuda()

    tout, tlogits = model1.forward_with_features(img)
    sout, slogits = model2.forward_with_features(img)

    # criterion_ce(tlogits, label).backward()
    # criterion_ce(slogits, label).backward()

    # tcompressed = model1.fc.weight.grad.unsqueeze(-1).unsqueeze(-1)
    # scompressed = model2.fc.weight.grad.unsqueeze(-1).unsqueeze(-1)

    visulize_sp_G(tout[-2].cpu().detach(),
                  sout[-2].cpu().detach(),
                  name=f'fig_of_{i}_th_ickd' if name is None else name)

    del model1, model2


def visualize_diswot_ickd_procedure(model1, model2, dataloader, name=None):
    network_weight_gaussian_init(model1)
    network_weight_gaussian_init(model2)
    model1.cuda()
    model2.cuda()

    dataiter = iter(dataloader)
    criterion_ce = nn.CrossEntropyLoss()

    try:
        img, label = next(dataiter)
        img, label = img.cuda(), label.cuda()
    except StopIteration as e:
        dataiter = iter(dataloader)
        img, label = next(dataiter)
        img, label = img.cuda(), label.cuda()

    tout, tlogits = model1.forward_with_features(img)
    sout, slogits = model2.forward_with_features(img)

    criterion_ce(tlogits, label).backward()
    criterion_ce(slogits, label).backward()

    tcompressed = model1.fc.weight.grad.unsqueeze(-1).unsqueeze(-1)
    scompressed = model2.fc.weight.grad.unsqueeze(-1).unsqueeze(-1)

    visulize_ickd_G(tcompressed.cpu().detach(),
                    scompressed.cpu().detach(),
                    name=f'fig_of_{i}_th_ickd' if name is None else name)

    del model1, model2


import torch.nn as nn


def ickd_with_grad_cam(teacher, student, batch_data):
    criterion = nn.CrossEntropyLoss()
    image, label = batch_data

    tlogits = teacher.forward(image)
    slogits = student.forward(image)

    criterion(tlogits, label).backward()
    criterion(slogits, label).backward()

    t_grad_cam = teacher.fc.weight.grad
    s_grad_cam = student.fc.weight.grad

    return -1 * ickd(t_grad_cam, s_grad_cam)


import torch.nn as nn


def sp_with_feature(teacher, student, batch_data):
    image, label = batch_data

    tfeats = teacher.forward_features(image)
    sfeats = student.forward_features(image)

    t_feat = tfeats[-2]
    s_feat = sfeats[-2]

    return -1 * ickd(t_feat, s_feat)


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
    tnet = get_teacher_best_model(TARGET, NUM_CLASSES)
    network_weight_gaussian_init(tnet)

    dataiter = iter(dataloader)
    img, label = next(dataiter)
    criterion_ce = nn.CrossEntropyLoss()
    # criterion_kl = nn.KLDivLoss()
    criterion_ickd = ICKDLoss()
    criterion_sp = Similarity()

    gt_list = []
    zcs_list = []

    chosen_idx = []
    for idx, dacc in j_file.items():
        # get random subnet
        snet, _ = get_network_by_index(int(idx))
        network_weight_gaussian_init(snet)
        gt_list.append(float(dacc))

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
        # visulize_ickd_G(tcompressed.detach(), scompressed.detach(), name=f'fig_of_{i}_th_subnet')

        score_sp = -1 * criterion_sp(tfeature[-1],
                                     sfeature[-1])[0].detach().numpy()
        score_ickd = -1 * criterion_ickd([tcompressed],
                                         [scompressed])[0].detach().numpy()
        result = score_sp + score_ickd

        zcs_list.append(result if isinstance(result, float) else result[0])

    print(
        f'kd: {kendalltau(zcs_list,gt_list):.4f} sp: {spearman(zcs_list,gt_list):.4f} ps: {pearson(zcs_list, gt_list):.4f}'
    )

    return spearman(zcs_list, gt_list)


def compute_zero_cost_proxies_rank_procedure(dataloader,
                                             zc_name,
                                             j_file=None):
    dataiter = iter(dataloader)
    img, label = next(dataiter)
    criterion_ce = nn.CrossEntropyLoss()

    gt_list = []
    zcs_list = []

    chosen_idx = []
    for k, v in j_file.items():
        # get random subnet
        snet, _ = get_network_by_index(int(k))
        network_weight_gaussian_init(snet)

        dataload_info = ['random', 3, NUM_CLASSES]
        gt_list.append(float(v))

        score = predictive.find_measures(snet,
                                         dataloader,
                                         dataload_info=dataload_info,
                                         measure_names=[zc_name],
                                         loss_fn=F.cross_entropy,
                                         device=torch.device('cpu'))

        zcs_list.append(score)

    print(
        f'{zc_name} kd: {kendalltau(zcs_list,gt_list):.4f} sp: {spearman(zcs_list,gt_list):.4f} ps: {pearson(zcs_list, gt_list):.4f}'
    )

    return kendalltau(zcs_list,
                      gt_list), spearman(zcs_list,
                                         gt_list), pearson(zcs_list, gt_list)


def compute_scda_kd_procedure(dataloader,
                              sample_number=200,
                              TARGET='cifar10',
                              NUM_CLASSES=10):
    # compare two random network with cifar100 data
    dataiter = iter(dataloader)
    img, _ = next(dataiter)
    criterion_rmi = RMIloss(img.size()[0])
    criterion_kl = nn.KLDivLoss()
    criterion_ickd = ICKDLoss()
    criterion_sp = Similarity()

    # tnet = resnet110(num_classes=100)
    tnet = get_teacher_best_model(TARGET, NUM_CLASSES)

    gt_list = []
    zcs_list = []

    chosen_idx = []
    for _ in tqdm(range(sample_number)):
        # get random subnet
        idx, snet, acc = random_sample_and_get_gt()
        if idx not in chosen_idx:
            chosen_idx.append(idx)
        else:
            idx, snet, acc = random_sample_and_get_gt()
        gt_list.append(acc)

        tfeature, tlogits = tnet.forward_with_features(img)
        sfeature, slogits = snet.forward_with_features(img)

        # compress depth wise information

        tcompressed = tfeature[-2]
        scompressed = sfeature[-2]

        # post process.
        zcs_list.append(
            -1 * criterion_sp(tcompressed, scompressed)[0].detach().numpy())

    print(
        f'kd: {kendalltau(zcs_list,gt_list)} sp: {spearman(zcs_list,gt_list)} ps: {pearson(zcs_list, gt_list)}'
    )


def main_kd_zc(TARGET, NUM_CLASSES):
    if TARGET == 'cifar100':
        train_loader, val_loader = get_cifar100_dataloaders(batch_size=128,
                                                            num_workers=0)
    elif TARGET == 'ImageNet16-120':
        train_loader, val_loader = get_imagenet16_dataloaders(batch_size=64,
                                                              num_workers=0)
    elif TARGET == 'cifar10':
        train_loader, _ = get_cifar10_dataloaders(batch_size=64, num_workers=0)
    # COMPUTE NORMAL KD RANK CONSISTENCY
    times = 10

    loss_dict = {}
    # loss_name_list = [
    #     'KD', 'FitNet', 'RKD', 'PKT', 'CC', 'AT', 'NST', 'KDSVD', 'SP'
    # ]
    loss_name_list = ['NST']
    for loss_name in loss_name_list:
        every_time_sp = []
        every_time_kd = []
        every_time_ps = []
        try:
            for _ in range(times):
                kd, sp, ps = compute_normal_kd_rank_procedure(
                    train_loader, loss_name, 50, TARGET, NUM_CLASSES)
                every_time_sp.append(sp)
                every_time_kd.append(kd)
                every_time_ps.append(ps)
            print(f'LOSS: {loss_name} LIST: {every_time_sp}')
            loss_dict[loss_name] = {
                'kd': every_time_kd,
                'sp': every_time_sp,
                'ps': every_time_ps,
            }
        except Exception:
            print(f"there is something wrong with {loss_name}")

def main_vanilla_zc(j_file, TARGET, NUM_CLASSES):
    if TARGET == 'cifar100':
        train_loader, val_loader = get_cifar100_dataloaders(batch_size=128,
                                                            num_workers=0)
    elif TARGET == 'ImageNet16-120':
        train_loader, val_loader = get_imagenet16_dataloaders(batch_size=64,
                                                              num_workers=0)
    elif TARGET == 'cifar10':
        train_loader, _ = get_cifar10_dataloaders(batch_size=64, num_workers=0)
    times = 10
    # COMPUTE ZC PROXIES RANK CONSISTENCY
    zc_dict = {}
    # zc_name_list = [
    #     'epe_nas', 'fisher', 'grad_norm', 'grasp', 'jacov', 'l2_norm', 'nwot',
    #     'plain', 'snip', 'synflow', 'flops', 'params', 'diswot', 'zen'
    # ]
    zc_name_list = ['flops', 'fisher', 'grad_norm', 'snip', 'synflow', 'nwot']

    for zc_name in zc_name_list:
        every_time_sp = []
        every_time_kd = []
        every_time_ps = []
        try:
            for _ in range(times):
                kd, sp, ps = compute_zero_cost_proxies_rank_procedure(
                    train_loader, zc_name, j_file)
                every_time_sp.append(sp)
                every_time_kd.append(kd)
                every_time_ps.append(ps)
            print(f'ZC: {zc_name} LIST: {every_time_sp}')
            zc_dict[zc_name] = {
                'kd': every_time_kd,
                'sp': every_time_sp,
                'ps': every_time_ps,
            }
        except Exception as e:
            print(f"there is something wrong with {zc_name} with {e}")



def main_plot_sp():
    from models.resnet_224 import resnet18, resnet34, resnet50, resnet101

    # train_loader, val_loader = get_cifar100_dataloaders(batch_size=64,
    # num_workers=0)
    train_loader, val_loader, n_data = get_cifar10_dataloaders_entropy(
        batch_size=32, num_workers=0)

    fig_name = f'./tmp/ResNet18-ResNet18-sp.png'
    visualize_diswot_sp_procedure(resnet18(pretrained=False),
                                  resnet18(pretrained=False),
                                  train_loader,
                                  name=fig_name)
    print('done')

    fig_name = f'./tmp/ResNet34-ResNet34-sp.png'
    visualize_diswot_sp_procedure(resnet34(pretrained=False),
                                  resnet34(pretrained=False),
                                  train_loader,
                                  name=fig_name)
    print('done')

    fig_name = f'./tmp/ResNet50-ResNet50-sp.png'
    visualize_diswot_sp_procedure(resnet50(pretrained=False),
                                  resnet50(pretrained=False),
                                  train_loader,
                                  name=fig_name)
    print('done')

    fig_name = f'./tmp/ResNet101-ResNet101-sp.png'
    visualize_diswot_sp_procedure(resnet101(pretrained=False),
                                  resnet101(pretrained=False),
                                  train_loader,
                                  name=fig_name)
    print('done')

    fig_name = f'./tmp/ResNet18-ResNet34-sp.png'
    visualize_diswot_sp_procedure(resnet18(pretrained=False),
                                  resnet34(pretrained=False),
                                  train_loader,
                                  name=fig_name)
    print('done')

    fig_name = f'./tmp/ResNet18-ResNet50-sp.png'
    visualize_diswot_sp_procedure(resnet18(pretrained=False),
                                  resnet50(pretrained=False),
                                  train_loader,
                                  name=fig_name)
    print('done')

    fig_name = f'./tmp/ResNet18-ResNet101-sp.png'
    visualize_diswot_sp_procedure(resnet18(pretrained=False),
                                  resnet101(pretrained=False),
                                  train_loader,
                                  name=fig_name)
    print('done')

    fig_name = f'./tmp/ResNet34-ResNet50-sp.png'
    visualize_diswot_sp_procedure(resnet34(pretrained=False),
                                  resnet50(pretrained=False),
                                  train_loader,
                                  name=fig_name)
    print('done')

    fig_name = f'./tmp/ResNet34-ResNet101-sp.png'
    visualize_diswot_sp_procedure(resnet34(pretrained=False),
                                  resnet101(pretrained=False),
                                  train_loader,
                                  name=fig_name)
    print('done')

    fig_name = f'./tmp/ResNet50-ResNet101-sp.png'
    visualize_diswot_sp_procedure(resnet50(pretrained=False),
                                  resnet101(pretrained=False),
                                  train_loader,
                                  name=fig_name)
    print('done')


def main_plot_ickd():
    from models.resnet_224 import resnet18, resnet34, resnet50, resnet101

    # train_loader, val_loader = get_cifar100_dataloaders(batch_size=64,
    #                                                     num_workers=0)
    train_loader, val_loader, n_data = get_cifar10_dataloaders_entropy(
        batch_size=32, num_workers=0)

    fig_name = f'./tmp/ResNet18-ResNet18-ickd.png'
    visualize_diswot_ickd_procedure(resnet18(pretrained=False),
                                    resnet18(pretrained=False),
                                    train_loader,
                                    name=fig_name)
    print('done')

    fig_name = f'./tmp/ResNet34-ResNet34-ickd.png'
    visualize_diswot_ickd_procedure(resnet34(pretrained=False),
                                    resnet34(pretrained=False),
                                    train_loader,
                                    name=fig_name)
    print('done')

    fig_name = f'./tmp/ResNet50-ResNet50-ickd.png'
    visualize_diswot_ickd_procedure(resnet50(pretrained=False),
                                    resnet50(pretrained=False),
                                    train_loader,
                                    name=fig_name)
    print('done')

    fig_name = f'./tmp/ResNet101-ResNet101-ickd.png'
    visualize_diswot_ickd_procedure(resnet101(pretrained=False),
                                    resnet101(pretrained=False),
                                    train_loader,
                                    name=fig_name)
    print('done')

    fig_name = f'./tmp/ResNet18-ResNet34-ickd.png'
    visualize_diswot_ickd_procedure(resnet18(pretrained=False),
                                    resnet34(pretrained=False),
                                    train_loader,
                                    name=fig_name)
    print('done')

    fig_name = f'./tmp/ResNet18-ResNet50-ickd.png'
    visualize_diswot_ickd_procedure(resnet18(pretrained=False),
                                    resnet50(pretrained=False),
                                    train_loader,
                                    name=fig_name)
    print('done')

    fig_name = f'./tmp/ResNet18-ResNet101-ickd.png'
    visualize_diswot_ickd_procedure(resnet18(pretrained=False),
                                    resnet101(pretrained=False),
                                    train_loader,
                                    name=fig_name)
    print('done')

    fig_name = f'./tmp/ResNet34-ResNet50-ickd.png'
    visualize_diswot_ickd_procedure(resnet34(pretrained=False),
                                    resnet50(pretrained=False),
                                    train_loader,
                                    name=fig_name)
    print('done')

    fig_name = f'./tmp/ResNet34-ResNet101-ickd.png'
    visualize_diswot_ickd_procedure(resnet34(pretrained=False),
                                    resnet101(pretrained=False),
                                    train_loader,
                                    name=fig_name)
    print('done')

    fig_name = f'./tmp/ResNet50-ResNet101-ickd.png'
    visualize_diswot_ickd_procedure(resnet50(pretrained=False),
                                    resnet101(pretrained=False),
                                    train_loader,
                                    name=fig_name)
    print('done')


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

    print(f'TARGET: {TARGET} / NUM_CLASSES: {NUM_CLASSES}')

    # arch_arr = sampling.nb201genostr2array(nb201_api.arch(sample))

    # train_loader, val_loader = get_cifar100_dataloaders(batch_size=64,
    #                                                     num_workers=0)

    # compute_scda_kd_procedure(train_loader)

    with open('./data/dist_nb201_bench.json', 'r') as f:
        j_file = json.load(f)

    # compute_diswot_procedure(50, TARGET, NUM_CLASSES)
    # compute_zen_1st_replace_grad_procedure()

    # from helper.utils.flop_benchmark import get_model_infos
    # tmodel = get_teacher_best_model(TARGET, NUM_CLASSES)
    # FLOPs, Param = get_model_infos(tmodel, shape=(1, 3, 32, 32))
    # print(f'FLOPS: {FLOPs} Param: {Param}')

    print('='*20)
    print('='*10, 'DISWOT')
    print('='*20)

    t1 = time.time()
    res_list = []
    for _ in range(5):
        res_list.append(compute_diswot_procedure(j_file, TARGET, NUM_CLASSES))
    print(res_list)
    print(f'====> time: {(time.time() - t1)/5} s')

    # main_plot_ickd()
    # main_plot_sp()

    # print('='*20)
    # print('='*10, 'KD')
    # print('='*20)

    main_kd_zc(TARGET, NUM_CLASSES)

    
    # print('='*20)
    # print('='*10, 'ZC under distill nb201 benchmark')
    # print('='*20)
    # main_vanilla_zc(j_file, TARGET, NUM_CLASSES)
