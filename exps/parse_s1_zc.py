import os
import sys
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rank_consisteny import kendalltau, pearson, spearman
from tqdm import tqdm

from dataset.cifar100 import get_cifar100_dataloaders
from distiller_zoo import ICKDLoss, RKDLoss, RMIloss, Similarity
from models import resnet56, resnet110, resnet110_zen
from models.candidates.mutable import mutable_resnet20, mutable_resnet20_zen
from predictor.pruners import predictive

plt.rc('font', family='Times New Roman')
GLOBAL_DPI = 600
FIGSIZE = (8, 6)
PADINCHES = 0.1  # -0.005
GLOBAL_FONTSIZE = 28
GLOBAL_LABELSIZE = 30
GLOBAL_LEGENDSIZE = 20

font1 = {
    'family': 'Times New Roman',
    'weight': 'bold',
    'size': GLOBAL_LABELSIZE
}

plt.rc('font', **font1)  # controls default text sizes
plt.rc('axes', titlesize=GLOBAL_LABELSIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=GLOBAL_LABELSIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=GLOBAL_LABELSIZE - 10)  # fontsize of the tick labels
plt.rc('ytick', labelsize=GLOBAL_LABELSIZE - 10)  # fontsize of the tick labels
plt.rc('legend', fontsize=GLOBAL_LEGENDSIZE)  # legend fontsize
plt.rc('figure', titlesize=GLOBAL_LABELSIZE)


def generate_config(mutable_depth=None):
    if mutable_depth is None:
        mutable_depth = [1, 3, 5, 7]
    config_list = []
    len_of_mutable = len(mutable_depth)

    for i in range(len_of_mutable):
        for j in range(len_of_mutable):
            for k in range(len_of_mutable):
                results = [
                    mutable_depth[i], mutable_depth[j], mutable_depth[k]
                ]
                config_list.append(results)
    return config_list


def visualize_1figures(gt_list: List,
                       zc_list: List,
                       title: str = None,
                       x_label='x',
                       y_label='y'):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)

    ax.scatter(gt_list, zc_list, s=100, marker='*', c='#3c73a8')
    # ax.set_title(f'{title}')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    plt.text(min(gt_list),
             min(zc_list) + (max(zc_list) - min(zc_list)) * 0.8,
             s=f"Kendall's Tau={kendalltau(gt_list, zc_list):.2f}",
             fontdict={
                 'fontsize': 20,
             })

    plt.grid(linestyle='-.', alpha=0.9, lw=2)
    plt.savefig(f'./{title}.png', dpi=600, bbox_inches='tight')
    # plt.show()


def train_snet_one_epoch(tnet, snet, train_loader):
    dataiter = iter(train_loader)
    img, _ = next(dataiter)

    criterion_ce = nn.CrossEntropyLoss()
    criterion_rmi = RMIloss(img.size()[0])
    for i, (img, label) in enumerate(train_loader):
        if i > 10:
            break
        sout, slogit = snet(img, is_feat=True, preact=False)
        tout, _ = tnet(img, is_feat=True, preact=False)
        loss = criterion_ce(slogit, label) * 0.2 + \
            criterion_rmi(tout, sout) * 0.8
        loss.backward()
    return loss.item()


def compute_rmi_score_procedure(kd_struct2acc):

    from dataset.cifar100 import get_cifar100_dataloaders_entropy
    train_loader, val_loader, n_data = get_cifar100_dataloaders_entropy()

    tnet = resnet110(num_classes=100)
    tpath = 'save/models/resnet110_vanilla/ckpt_epoch_240.pth'
    tnet.load_state_dict(torch.load(tpath)['model'])

    dataiter = iter(train_loader)
    img, _ = next(dataiter)
    # tout, _ = tnet(img, is_feat=True, preact=False)
    # sout = snet(img, is_feat=True, preact=False)

    gt_list = []
    rmi_list = []

    for struct, acc in tqdm(kd_struct2acc.items()):
        gt_list.append(acc)
        depth_list = [int(item) for item in list(struct)]
        snet = mutable_resnet20(depth_list, num_classes=100)
        tout, _ = tnet(img, is_feat=True, preact=False)
        sout, _ = snet(img, is_feat=True, preact=False)
        loss = train_snet_one_epoch(tnet, snet, train_loader)
        rmi_list.append(loss)
        del snet
        del tout
        del sout

    print(
        f'kd: {kendalltau(rmi_list,gt_list)} sp: {spearman(rmi_list,gt_list)} ps: {pearson(rmi_list, gt_list)}'
    )


def compute_cam_procedure(kd_struct2acc, dataloader):
    # cam not grad-cam
    tnet = resnet110(num_classes=100)

    # fc_weight = tnet.fc.weight
    # cam_weight = fc_weight.unsqueeze(-1).unsqueeze(-1)

    dataiter = iter(dataloader)
    img, _ = next(dataiter)
    criterion_rmi = RMIloss(img.size()[0])
    criterion_kl = nn.KLDivLoss()
    criterion_ickd = ICKDLoss()

    tnet = resnet110(num_classes=100)

    gt_list = []
    diswot_list = []

    for struct, acc in tqdm(kd_struct2acc.items()):
        gt_list.append(acc)
        depth_list = [int(item) for item in list(struct)]
        snet = mutable_resnet20(depth_list, num_classes=100)
        tnet(img, is_feat=True, preact=False)
        snet(img, is_feat=True, preact=False)

        # compress depth wise information
        # tcompressed = tout[-1].unsqueeze(-1).unsqueeze(-1)
        # scompressed = sout[-1].unsqueeze(-1).unsqueeze(-1)
        tcompressed = tnet.fc.weight.unsqueeze(-1).unsqueeze(-1)
        scompressed = snet.fc.weight.unsqueeze(-1).unsqueeze(-1)

        # post process.
        # diswot_list.append(-(torch.sum(tcompressed - scompressed).detach().numpy()))
        # diswot_list.append(criterion_kl(tcompressed, scompressed).detach().numpy())
        diswot_list.append(
            torch.mean(F.cosine_similarity(tcompressed,
                                           scompressed)).detach().numpy())
        # diswot_list.append(criterion_rmi(tcompressed, scompressed).detach().numpy())
        # diswot_list.append(-1 * criterion_ickd([tcompressed], [scompressed])[0].detach().numpy())

    print(
        f'kd: {kendalltau(diswot_list,gt_list)} sp: {spearman(diswot_list,gt_list)} ps: {pearson(diswot_list, gt_list)}'
    )


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


def compute_zen_1st_procedure(kd_struct2acc, dataloader):
    # grad-cam not cam
    tnet = resnet110(num_classes=100)
    # tnet = resnet56(num_classes=100)
    network_weight_gaussian_init(tnet)

    device = torch.device('cpu')
    dtype = torch.float32
    mixup_gamma = 1e-2

    # inputs
    input = torch.randn(size=(128, 3, 32, 32), device=device, dtype=dtype)
    input2 = torch.randn(size=(128, 3, 32, 32), device=device, dtype=dtype)
    mixup_input = input + mixup_gamma * input2

    tnet = tnet.to(device)

    dataiter = iter(dataloader)
    img, label = next(dataiter)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_rmi = RMIloss(img.size()[0])
    criterion_kl = nn.KLDivLoss()
    criterion_ickd = ICKDLoss()
    criterion_sp = Similarity()

    gt_list = []
    diswot_list = []

    for struct, acc in tqdm(kd_struct2acc.items()):
        gt_list.append(acc)
        depth_list = [int(item) for item in list(struct)]
        snet = mutable_resnet20(depth_list, num_classes=100)
        network_weight_gaussian_init(snet)

        tfeature1, _ = tnet(input, is_feat=True, preact=False)
        tfeature2, _ = tnet(mixup_input, is_feat=True, preact=False)
        tscore_tensor = torch.abs(tfeature1[-2] - tfeature2[-2])

        sfeature1, _ = snet(input, is_feat=True, preact=False)
        sfeature2, _ = snet(mixup_input, is_feat=True, preact=False)
        sscore_tensor = torch.abs(sfeature1[-2] - sfeature2[-2])

        tcompressed = tscore_tensor  # .unsqueeze(-1).unsqueeze(-1)
        scompressed = sscore_tensor  # .unsqueeze(-1).unsqueeze(-1)

        # post process.
        # diswot_list.append(-(torch.sum(tcompressed - scompressed).detach().numpy()))
        # diswot_list.append(-1 * criterion_kl(tcompressed, scompressed).detach().numpy()) # 18
        # diswot_list.append(torch.mean(F.cosine_similarity(tcompressed, scompressed)).detach().numpy())
        # diswot_list.append(criterion_rmi(tcompressed, scompressed).detach().numpy())
        # diswot_list.append(-1 * criterion_ickd([tcompressed], [scompressed])[0].detach().numpy())
        # diswot_list.append(-1 * criterion_sp(tcompressed, scompressed)[0].detach().numpy()) # 23

        # -1 * criterion_kl(tcompressed, scompressed).detach().numpy() + \
        diswot_list.append(
            -1 *
            criterion_ickd([tcompressed], [scompressed])[0].detach().numpy() +
            -1 *
            criterion_sp(tfeature1[-2], tfeature1[-2])[0].detach().numpy())

    print(
        f'kd: {kendalltau(diswot_list,gt_list)} sp: {spearman(diswot_list,gt_list)} ps: {pearson(diswot_list, gt_list)}'
    )


def compute_grad_cam_procedure(kd_struct2acc, dataloader):
    # grad-cam not cam
    # tnet = resnet110(num_classes=100)

    tnet = resnet56(num_classes=100)
    network_weight_gaussian_init(tnet)

    dataiter = iter(dataloader)
    img, label = next(dataiter)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_rmi = RMIloss(img.size()[0])
    criterion_kl = nn.KLDivLoss()
    criterion_ickd = ICKDLoss()
    criterion_sp = Similarity()
    criterion_rkd = RKDLoss()
    criterion_hint = nn.MSELoss()

    gt_list = []
    diswot_list = []
    struct_list = []

    for struct, acc in tqdm(kd_struct2acc.items()):
        gt_list.append(acc)
        struct_list.append(struct)

        depth_list = [int(item) for item in list(struct)]
        snet = mutable_resnet20(depth_list, num_classes=100)
        network_weight_gaussian_init(snet)

        tout, tlogits = tnet(img, is_feat=True, preact=False)
        sout, slogits = snet(img, is_feat=True, preact=False)

        criterion_ce(tlogits, label).backward()
        criterion_ce(slogits, label).backward()

        # # fc.weight.grad
        # t_fc_weight = tnet.fc.weight.grad
        # t_cam_weight = torch.index_select(t_fc_weight, 0, tlogits.argmax(dim=1)).unsqueeze(-1).unsqueeze(-1)

        # s_fc_weight = snet.fc.weight.grad
        # s_cam_weight = torch.index_select(s_fc_weight, 0, slogits.argmax(dim=1)).unsqueeze(-1).unsqueeze(-1)

        # diswot_list.append(-1 * criterion_hint(tout[-2], sout[-2]).detach().numpy())
        # diswot_list.append(-1 * criterion_rkd(tout[-2], sout[-2]).detach().numpy())

        # tcompressed = t_cam_weight
        # scompressed = s_cam_weight

        # tcompressed = tnet.fc.weight.grad.unsqueeze(-1).unsqueeze(-1)
        # scompressed = snet.fc.weight.grad.unsqueeze(-1).unsqueeze(-1)

        # post process.
        # diswot_list.append(-(torch.sum(tcompressed - scompressed).detach().numpy()))
        # diswot_list.append(-1 * criterion_kl(tcompressed, scompressed).detach().numpy()) # 18
        # diswot_list.append(torch.mean(F.cosine_similarity(tcompressed, scompressed)).detach().numpy())
        # diswot_list.append(criterion_rmi(tcompressed, scompressed).detach().numpy())
        # diswot_list.append(-1 * criterion_ickd([tcompressed], [scompressed])[0].detach().numpy())
        # 23
        # diswot_list.append(-1 * criterion_sp(tout[-2], sout[-2])[0].detach().numpy()[0])
        # item = -1 * criterion_ickd([tcompressed], [scompressed])[0].detach().numpy()
        # item1 = -1 * criterion_kl(tcompressed, scompressed).detach().numpy() # 71
        # item = -1 * criterion_sp(tout[-2], sout[-2])[0].detach().numpy() + -1 * criterion_ickd([tcompressed], [scompressed])[0].detach().numpy()
        # + -1 * criterion_kl(tcompressed, scompressed).detach().numpy() +
        # item = -1 * criterion_ickd([tcompressed], [scompressed])[0].detach().numpy()
        # item1 = torch.clamp(item1, min=1e-7, max=1 - 1e-7)
        # item2 = torch.clamp(item2, min=1e-7, max=1 - 1e-7)
        # result = item1 + item2
        # diswot_list.append(item)
        # diswot_list.append(-1 * criterion_ickd([tout[-5]], [sout[-5]])[0].detach().numpy())
        # diswot_list.append(-1 * criterion_sp(tcompressed, scompressed)[0].detach().numpy()[0])
        print(diswot_list)

    print(
        f'kd: {kendalltau(diswot_list,gt_list)} sp: {spearman(diswot_list,gt_list)} ps: {pearson(diswot_list, gt_list)}'
    )

    info_dict = {
        s: {
            'gt': g,
            'pre': p
        }
        for s, g, p in zip(struct_list, gt_list, diswot_list)
    }
    return kendalltau(diswot_list,
                      gt_list), spearman(diswot_list, gt_list), pearson(
                          diswot_list, gt_list), info_dict

    # import numpy as np
    # print(f'real: {np.argsort(gt_list).tolist()}')
    # print(f'zero: {np.argsort(diswot_list).tolist()}')


def compute_diswot_kd_procedure(kd_struct2acc, dataloader):
    # compare two random network with cifar100 data
    dataiter = iter(dataloader)
    img, label = next(dataiter)
    criterion_rmi = RMIloss(img.size()[0])
    criterion_kl = nn.KLDivLoss()
    criterion_ickd = ICKDLoss()
    criterion_sp = Similarity()
    criterion_ce = nn.CrossEntropyLoss()

    tnet = resnet110(num_classes=100)
    network_weight_gaussian_init(tnet)

    gt_list = []
    diswot_list = []

    for struct, acc in tqdm(kd_struct2acc.items()):
        gt_list.append(acc)
        depth_list = [int(item) for item in list(struct)]
        snet = mutable_resnet20(depth_list, num_classes=100)
        network_weight_gaussian_init(snet)
        tout, tlogits = tnet(img, is_feat=True, preact=False)
        sout, slogits = snet(img, is_feat=True, preact=False)

        # compress depth wise information
        # tcompressed = tout[-1].unsqueeze(-1).unsqueeze(-1)
        # scompressed = sout[-1].unsqueeze(-1).unsqueeze(-1)
        # tcompressed = tout[-2]
        # scompressed = sout[-2]

        criterion_ce(tlogits, label).backward()
        criterion_ce(slogits, label).backward()
        tcompressed = tnet.fc.weight.grad.unsqueeze(-1).unsqueeze(-1)
        scompressed = snet.fc.weight.grad.unsqueeze(-1).unsqueeze(-1)

        # post process.
        # diswot_list.append(-(torch.sum(tcompressed - scompressed).detach().numpy()))
        # diswot_list.append(-1 * criterion_kl(tcompressed, scompressed).detach().numpy())
        # diswot_list.append(torch.mean(F.cosine_similarity(tcompressed, scompressed)).detach().numpy())
        # diswot_list.append(criterion_rmi(tcompressed, scompressed).detach().numpy())

        # result = -1 * criterion_sp(tout[-2], sout[-2])[0].detach().numpy() +
        result = -1 * criterion_ickd(tout[-2], sout[-2])[0].detach().numpy()
        # + -1 * criterion_kl(tcompressed, scompressed).detach().numpy()
        diswot_list.append(result)

        # diswot_list.append(-1 * criterion_ickd([tcompressed], [scompressed])[0].detach().numpy())
        # diswot_list.append(-1 * criterion_sp(tcompressed, scompressed)[0].detach().numpy())

    print(gt_list)
    print(diswot_list)

    visualize_1figures(
        gt_list,
        diswot_list,
        title=
        f'Correlation of Vanilla Acc. and DisWOT Score_{kendalltau(diswot_list,gt_list):.2f}',
        x_label='Vanilla Acc.',
        y_label='DisWOT Score')

    print(
        f'kd: {kendalltau(diswot_list,gt_list)} sp: {spearman(diswot_list,gt_list)} ps: {pearson(diswot_list, gt_list)}'
    )


def compute_zc_kd_procedure(kd_struct2acc, val_loader, zc='nwot'):
    if isinstance(zc, str):
        zc = [zc]
    dataload_info = ['random', 3, 100]
    # process kd_struct2acc
    gt_list = []
    diswot_list = []

    for struct, acc in kd_struct2acc.items():
        depth_list = [int(item) for item in list(struct)]
        snet = mutable_resnet20_zen(depth_list, num_classes=100)
        network_weight_gaussian_init(snet)
        sscore = predictive.find_measures(snet,
                                          val_loader,
                                          dataload_info=dataload_info,
                                          measure_names=zc,
                                          loss_fn=F.cross_entropy,
                                          device=torch.device('cuda:0'))
        gt_list.append(acc)
        diswot_list.append(sscore)

    print(f'kd of {zc} is {kendalltau(gt_list, diswot_list)} / sp of {zc} is {spearman(gt_list, diswot_list)} / ps of {zc} is {pearson(gt_list, diswot_list)}')

    visualize_1figures(
        gt_list,
        diswot_list,
        title=
        f'Correlation of Distill Acc. and NWOT Score_{kendalltau(diswot_list,gt_list):.2f}',
        x_label='Distill Acc.',
        y_label='NWOT Score')
    return spearman(gt_list, diswot_list)


def compute_zc_cls_procedure(cls_struct2acc, val_loader, zc='nwot'):
    if isinstance(zc, str):
        zc = [zc]
    dataload_info = ['random', 3, 100]
    # process cls_struct2acc
    print('process cls_struct2acc...')
    gt_list = []
    zc_list1 = []

    for struct, acc in tqdm(cls_struct2acc.items()):
        depth_list = [int(item) for item in list(struct)]
        snet = mutable_resnet20(depth_list, num_classes=100)
        sscore = predictive.find_measures(snet,
                                          val_loader,
                                          dataload_info=dataload_info,
                                          measure_names=zc,
                                          loss_fn=F.cross_entropy,
                                          device=torch.device('cuda:0'))
        gt_list.append(acc)
        zc_list1.append(sscore)

    print(
        f'kd: {kendalltau(zc_list1,gt_list)} sp: {spearman(zc_list1,gt_list)} ps: {pearson(zc_list1, gt_list)}'
    )

    visualize_1figures(gt_list,
                       zc_list1,
                       title='Correlation of Vanilla Acc. and NWOT Score',
                       x_label='Vanilla Acc.',
                       y_label='NWOT Score')


def compute_kd_cls2kd(cls_struct2acc, kd_struct2acc):
    # kendall's between cls and kd
    list_cls = []
    list_kd = []
    for (k1, v1), (k2, v2) in zip(cls_struct2acc.items(),
                                  kd_struct2acc.items()):
        list_cls.append(v1)
        list_kd.append(v2)
    kd = kendalltau(list_cls, list_kd)
    print(kd)


if __name__ == '__main__':
    # load cls benchmark
    cls_struct2acc = dict()
    with open('./exps/s1-gt-cls.txt', 'r') as f:
        contents = f.readlines()
        for content in contents:
            struct, acc = content.split()
            acc = float(acc)
            cls_struct2acc[struct] = acc

    # load kd benchmark
    kd_struct2acc = dict()
    with open('./exps/s1-gt-kd.txt', 'r') as f:
        contents = f.readlines()
        for content in contents:
            struct, acc = content.split()
            acc = float(acc)
            kd_struct2acc[struct] = acc

    config_list = generate_config()

    train_loader, val_loader = get_cifar100_dataloaders(batch_size=128,
                                                        num_workers=2,
                                                        is_instance=False)

    # supplimentary fig1
    # a = []
    # b = []
    # for k, v in kd_struct2acc.items():
    #     a.append(v)
    #     b.append(cls_struct2acc[k])
    # visualize_1figures(b, a,
    #                    title='Correlation of Vanilla Acc. and Distill Acc.',
    #                    x_label='Vanilla Acc.',
    #                    y_label='Distill Acc.')
    # print("done")

    # supp. fig2
    # compute_diswot_kd_procedure(kd_struct2acc, train_loader)

    # supp. fig3
    # compute_zc_cls_procedure(cls_struct2acc=cls_struct2acc,
    #                          val_loader=val_loader,
    #                          zc='nwot')

    # supp. fig4
    # compute_zc_kd_procedure(kd_struct2acc=kd_struct2acc,
    #                         val_loader=val_loader,
    #                         zc='nwot')

    # ===== available zc choices
    zc_list = ['epe_nas', 'fisher', 'grad_norm', 'grasp', 'jacov', 'l2_norm',
               'nwot', 'plain', 'snip', 'synflow', 'flops', 'params', 'zen']
    # zc_list = ['nwot']
    # zc_list = ['l2_norm', 'nwot', 'plain', 'flops', 'params', 'zen', 'flops', 'params', 'zen']
    times = 10
    for zc in zc_list:
        t = []
        for _ in range(times):
            sp = compute_zc_kd_procedure(kd_struct2acc=kd_struct2acc,
                                val_loader=val_loader,
                                zc=zc)
            t.append(sp)
        print(f"current zc: {zc} result list: {t} mean: {np.mean(t)} std: {np.std(t)}")
        # compute_zc_cls_procedure(cls_struct2acc=cls_struct2acc,
        #                          val_loader=val_loader,
        #                          zc=zc)

    # # compute_rmi_score_procedure(kd_struct2acc)
    # compute_cam_procedure(kd_struct2acc, train_loader)

    # result_list = []
    # for _ in range(10):
    #     r = compute_grad_cam_procedure(kd_struct2acc, train_loader)
    #     result_list.append(r)
    # for item in result_list:
    #     print(item)

    # compute_zen_1st_procedure(kd_struct2acc, train_loader)

    # _, _, _, info = compute_grad_cam_procedure(kd_struct2acc, train_loader)
    # import json
    # with open('diswot_sp_score_info_v2.txt', 'w') as f:
    #     f.write(json.dumps(info))
