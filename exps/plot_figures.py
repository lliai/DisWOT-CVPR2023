import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import ticker

plt.rc('font', family='Times New Roman')
GLOBAL_DPI = 600
FIGSIZE = (8, 6)
PADINCHES = 0.1  #-0.005
GLOBAL_FONTSIZE = 34
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

ax = plt.gca()
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)


def plot_standalone_model_rank(xlabel, ylabel, foldername, file_name):
    dpi, width, height = GLOBAL_DPI, 4000, 1350
    figsize = width / float(dpi), height / float(dpi)

    fig, axs = plt.subplots(1, 3, figsize=figsize)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none',
                    top=False,
                    bottom=False,
                    left=False,
                    right=False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ticks = np.arange(0, 51, 10)
    axs[0].set_xticks(ticks)
    axs[0].set_xticklabels(ticks)
    axs[1].set_xticks(ticks)
    axs[1].set_xticklabels(ticks)
    axs[2].set_xticks(ticks)
    axs[2].set_xticklabels(ticks)
    axs[0].set_xticks(ticks)
    axs[0].set_yticklabels(ticks)
    axs[1].set_yticks(ticks)
    axs[1].set_yticklabels(ticks)
    axs[2].set_yticks(ticks)
    axs[2].set_yticklabels(ticks)
    for ax in axs.flat:
        ax.label_outer()

    ax1 = axs[0]
    real_rank = [
        0, 16, 32, 48, 4, 20, 36, 52, 8, 24, 1, 40, 12, 56, 17, 28, 33, 44, 21,
        5, 60, 49, 37, 2, 53, 25, 9, 50, 13, 18, 3, 34, 57, 29, 61, 41, 6, 45,
        10, 22, 35, 38, 19, 26, 7, 58, 54, 46, 39, 14, 30, 51, 23, 62, 42, 55,
        11, 15, 59, 31, 27, 63, 47, 43
    ]
    angle_rank = [
        28, 0, 12, 36, 32, 16, 24, 20, 8, 60, 4, 52, 40, 48, 44, 56, 9, 5, 21,
        17, 13, 1, 29, 25, 61, 37, 53, 41, 33, 57, 49, 30, 45, 10, 2, 6, 18,
        14, 46, 26, 7, 38, 34, 22, 35, 27, 47, 23, 42, 19, 11, 3, 39, 54, 50,
        15, 31, 58, 51, 59, 63, 62, 55, 43
    ]

    ax1.scatter(real_rank, angle_rank, alpha=0.6)
    ax1.set_title('CIFAR-10')
    ax1.text(real_rank[-1] - 23, 2, 'Tau=0.710', fontsize=20)
    ax1.set_xticks(np.arange(0, real_rank[-1] + 1, 10))
    ax1.set_yticks(np.arange(0, real_rank[-1] + 1, 10))

    ax2 = axs[1]

    real_rank = [
        31, 38, 3, 15, 47, 30, 0, 37, 44, 9, 36, 19, 10, 16, 45, 13, 35, 41,
        25, 1, 28, 14, 22, 18, 48, 46, 42, 33, 32, 26, 40, 34, 24, 23, 5, 12,
        6, 17, 4, 20, 7, 11, 8, 43, 29, 39, 49, 27, 2, 21
    ]
    angle_rank = [
        31, 15, 9, 25, 30, 35, 14, 38, 13, 10, 16, 1, 3, 22, 40, 32, 47, 37,
        46, 48, 11, 5, 2, 42, 45, 0, 23, 34, 18, 8, 26, 6, 12, 33, 41, 36, 19,
        17, 4, 29, 21, 44, 27, 43, 20, 7, 39, 24, 28, 49
    ]

    ax2.scatter(real_rank, angle_rank, alpha=0.6)
    ax2.set_title('CIFAR-100')
    ax2.text(real_rank[-1] - 23, 2, 'Tau=0.471', fontsize=20)
    ax2.set_xticks(np.arange(0, real_rank[-1] + 1, 10))
    ax2.set_yticks(np.arange(0, real_rank[-1] + 1, 10))

    ax3 = axs[2]
    real_rank = [
        28, 1, 22, 37, 39, 19, 27, 25, 33, 8, 29, 17, 24, 49, 21, 20, 44, 4, 5,
        6, 40, 16, 12, 45, 9, 18, 36, 23, 38, 15, 46, 43, 41, 0, 10, 35, 3, 11,
        2, 26, 32, 13, 14, 34, 7, 30, 31, 48, 42, 47
    ]
    angle_rank = [
        17, 1, 28, 29, 25, 22, 27, 35, 39, 20, 19, 38, 11, 44, 0, 15, 8, 12, 3,
        37, 32, 9, 36, 41, 46, 47, 2, 30, 23, 13, 18, 24, 7, 14, 49, 33, 40, 4,
        21, 48, 31, 34, 26, 10, 16, 6, 5, 45, 42, 43
    ]
    ax3.scatter(real_rank, angle_rank, alpha=0.6)
    ax3.set_title('ImageNet-16-120')  # 0.851
    ax3.text(real_rank[-1] - 23, 2, 'Tau=0.324', fontsize=20)
    ax3.set_xticks(np.arange(0, real_rank[-1] + 1, 10))
    ax3.set_yticks(np.arange(0, real_rank[-1] + 1, 10))
    fig.tight_layout()

    save_path = os.path.join(foldername, file_name)
    # foldername / '{}'.format(file_name)
    print('save figure into {:}\n'.format(save_path))
    plt.savefig(str(save_path),
                dpi=GLOBAL_DPI,
                bbox_inches='tight',
                pad_inches=PADINCHES,
                format='pdf')
    plt.clf()


def plot_ranking_stability(xlabel, ylabel, foldername, file_name):
    plt.figure(figsize=FIGSIZE)

    ickd_data = []
    diswot_data = []
    sp_data = []
    datasets = ['cifar10', 'cifar100', 'ImageNet16-120']

    for dataset in datasets:
        if dataset == 'cifar10':
            # random_data.append([random.random() for _ in range(10)])
            # acc_no_rebn_data.append([random.random() for _ in range(10)])
            ickd_data.append([
                0.2434, 0.5166, 0.4982, 0.3267, 0.4508, 0.4443, 0.4303, 0.4067,
                0.4822, 0.5768
            ])
            diswot_data.append([
                0.4802, 0.4044, 0.4315, 0.4877, 0.4142, 0.5253, 0.5304, 0.3731,
                0.5994, 0.4575
            ])
            # kendall's tau
            sp_data.append([
                0.5264, 0.3642, 0.3381, 0.4338, 0.5172, 0.4232, 0.5474, 0.3861,
                0.2098, 0.4706
            ])

    width = 0.20
    locations = list(range(len(ickd_data)))
    locations = [i + 1 - 0.135 for i in locations]

    positions1 = locations
    boxplot1 = plt.boxplot(diswot_data,
                           positions=positions1,
                           patch_artist=True,
                           showfliers=True,
                           widths=width)

    positions2 = [x + (width + 0.08) for x in locations]
    boxplot2 = plt.boxplot(ickd_data,
                           positions=positions2,
                           patch_artist=True,
                           showfliers=True,
                           widths=width)

    positions3 = [x + (width + 0.08) * 2 for x in locations]
    boxplot3 = plt.boxplot(sp_data,
                           positions=positions3,
                           patch_artist=True,
                           showfliers=True,
                           widths=width)

    for box in boxplot1['boxes']:
        box.set(color='#3c73a8')

    for box in boxplot2['boxes']:
        box.set(color='#fec615')

    for box in boxplot3['boxes']:
        box.set(color='#2ec615')

    plt.xlim(0, len(ickd_data) + 1)

    ticks = np.arange(0, len(ickd_data) + 1, 1)
    ticks_label_ = ['CIFAR-10', 'CIFAR-100', 'ImageNet-16-120', '']
    ticks_label = []

    for i in range(len(ickd_data) + 1):
        ticks_label.append(str(ticks_label_[i - 1]))
    plt.xticks(ticks, ticks_label)  # , rotation=45)
    plt.xlabel(xlabel, fontsize=25)
    plt.ylabel(ylabel, fontsize=25)

    plt.grid(lw=2, ls='-.')
    plt.plot([], c='#3c73a8', label='Acc. w/ ReBN')
    plt.plot([], c='#fec615', label='Angle')
    plt.legend(ncol=6, loc='lower center', bbox_to_anchor=(0.5, -0.55))

    save_path = os.path.join(foldername, file_name)

    print('save figure into {:}\n'.format(save_path))
    plt.savefig(str(save_path),
                bbox_inches='tight',
                dpi=GLOBAL_DPI,
                pad_inches=PADINCHES,
                format='pdf')
    plt.clf()


def plot_evolution_search_process():
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

    x = []
    y = []
    with open('./exps/evo_search_img.log', 'r') as f:
        contents = f.readlines()
        for i, c in enumerate(contents):
            max_score = float(c.split(', ')[1].split('=')[1])
            x.append(i)
            y.append(max_score)

    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)

    plt.grid(linestyle='-.', lw=2, alpha=0.9)
    # print(x, y)
    plt.plot(x,
             y,
             color='salmon',
             linestyle='-',
             lw=3,
             label='Evolution search')

    x2, y2 = [], []
    with open('./exps/rand_search_img.log', 'r') as f:
        contents = f.readlines()
        for i, c in enumerate(contents):
            max_score = float(c.split(', ')[1].split('=')[1])
            x2.append(i)
            y2.append(max_score)

    plt.plot(x2,
             y2,
             color='skyblue',
             linestyle='-',
             lw=4,
             label='Random search')

    plt.legend()
    xticks = np.arange(0, 1000, 250)
    plt.ylim([-0.00053, -0.00018])
    plt.xticks(xticks, fontsize=GLOBAL_LABELSIZE)
    plt.tick_params(labelsize=GLOBAL_LEGENDSIZE)
    plt.xlabel('Iteration', fontsize=GLOBAL_LEGENDSIZE, weight='bold')
    plt.ylabel('DisWOT Metric', font1)

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter)

    plt.tight_layout()
    plt.savefig('./tmp/ES_vs_RS.pdf',
                dpi=GLOBAL_DPI,
                bbox_inches='tight',
                pad_inches=PADINCHES,
                format='pdf')
    plt.clf()


def plot_kd_zc_box(foldername, file_name):
    import json
    plt.figure(figsize=FIGSIZE)

    # load from files
    kd_file_path = './exps/kd_name_results_x10.txt'
    zc_file_path = './exps/zc_name_results_x10.txt'
    with open(kd_file_path, 'r') as f:
        kd_info = json.load(f)
    with open(zc_file_path, 'r') as f:
        zc_info = json.load(f)
    # merge kd and zc
    merged_info = {**kd_info, **zc_info}

    total_boxes = len(merged_info)

    plt.figure(figsize=(32, 4))

    width = 0.20
    locations = list(range(total_boxes))
    locations = [i + 1 - 0.135 for i in locations]

    labels = []
    content_list = []
    for i, (k, v) in enumerate(merged_info.items()):
        labels.append(k)
        content_list.append(v)

    plt.boxplot(content_list,
                medianprops={
                    'color': 'red',
                    'linewidth': '1.5'
                },
                meanline=True,
                showmeans=True,
                meanprops={
                    'color': 'blue',
                    'ls': '-.',
                    'linewidth': '1.5'
                },
                flierprops={
                    'marker': 'o',
                    'markerfacecolor': 'red',
                    'markersize': 10
                },
                labels=labels)

    plt.xlim(0, total_boxes + 1)

    plt.grid(lw=2, ls='-.')
    plt.tight_layout()

    save_path = os.path.join(foldername, file_name)

    print('save figure into {:}\n'.format(save_path))
    plt.savefig(str(save_path),
                bbox_inches='tight',
                dpi=GLOBAL_DPI,
                pad_inches=PADINCHES,
                format='pdf')
    plt.clf()


def plot_kd_box(foldername, file_name):
    import json
    plt.figure(figsize=FIGSIZE)
    # load from files
    kd_file_path = './exps/kd_name_results_x10.txt'
    with open(kd_file_path, 'r') as f:
        kd_info = json.load(f)

    for k, v in kd_info.items():
        print(k, np.mean(np.array(v)))

    # merge kd and zc
    merged_info = {**kd_info}

    total_boxes = len(merged_info)
    plt.figure(figsize=FIGSIZE)
    ax = plt.gca()
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)

    locations = list(range(total_boxes))
    locations = [i + 1 - 0.135 for i in locations]

    labels = []
    content_list = []
    for i, (k, v) in enumerate(merged_info.items()):
        labels.append(k)
        content_list.append(v)

    plt.boxplot(content_list,
                medianprops={
                    'color': 'red',
                    'linewidth': '2'
                },
                meanline=True,
                showmeans=True,
                meanprops={
                    'color': 'blue',
                    'ls': '-.',
                    'linewidth': '2'
                },
                flierprops={
                    'marker': 'o',
                    'markerfacecolor': 'red',
                    'markersize': 20
                },
                labels=labels,
                boxprops={'linewidth': '2'},
                whiskerprops={'linewidth': '2'},
                widths=0.8)

    plt.xlim(0, total_boxes + 1)

    plt.grid(lw=2, ls='-.')
    plt.tight_layout()

    save_path = os.path.join(foldername, file_name)
    plt.ylabel('Spearman Coefficienct',
               fontsize=GLOBAL_LABELSIZE,
               weight='bold')
    plt.xlabel('Distillation Method',
               fontsize=GLOBAL_LEGENDSIZE,
               weight='bold')

    plt.tick_params(labelsize=GLOBAL_LEGENDSIZE - 4)

    print('save figure into {:}\n'.format(save_path))
    plt.savefig(str(save_path),
                bbox_inches='tight',
                dpi=GLOBAL_DPI,
                pad_inches=PADINCHES,
                format='pdf')
    plt.clf()


def plot_different_teacher_diswot():
    plt.figure(figsize=FIGSIZE)
    labels = ['ResNet32', 'ResNet44', 'ResNet56', 'ResNet110']
    res20_list = [70.24, 70.56, 70.98, 70.79]
    diswot_list = [71.01, 71.25, 71.63, 71.84]
    diswot_plus_list = [71.85, 72.12, 72.56, 72.92]
    marker_size = 13

    ax = plt.gca()
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)

    plt.grid(linestyle='-.', alpha=0.9, lw=2)
    plt.plot(
        labels,
        res20_list,
        color='#FFBE7A',
        # mfc='white',
        linewidth=2,
        marker='o',
        # linestyle=':',
        label='KD',
        markersize=marker_size)
    plt.plot(
        labels,
        diswot_list,
        color='#2878B5',
        # mfc='white',
        linewidth=2,
        marker='v',
        markersize=marker_size,
        label='DisWOT')
    plt.plot(
        diswot_plus_list,
        color='#32B897',
        # mfc='white',
        linewidth=2,
        marker='*',
        markersize=marker_size + 5,
        label='DisWOT+')
    plt.tick_params(labelsize=GLOBAL_LEGENDSIZE)
    plt.xlabel('Teacher Models', fontsize=GLOBAL_LEGENDSIZE, weight='bold')
    plt.ylabel('Top-1 Accuracy (%)', fontsize=GLOBAL_FONTSIZE, weight='bold')
    label_size = plt.legend().get_texts()
    [label.set_fontsize(GLOBAL_LEGENDSIZE - 6) for label in label_size]
    plt.tight_layout()
    plt.savefig('./tmp/diff_teacher_diswot.pdf',
                dpi=GLOBAL_DPI,
                bbox_inches='tight',
                pad_inches=PADINCHES,
                format='pdf')
    plt.clf()


def plot_hist_rank_consistency():
    result = {
        'Fisher': {
            'cls': 0.81,
            'kd': 0.63
        },
        'NWOT': {
            'cls': 0.40,
            'kd': 0.32
        },
        'SNIP': {
            'cls': 0.85,
            'kd': 0.67
        },
        'Vanilla acc.': {
            'cls': 1.00,
            'kd': 0.85
        }
    }
    # result = {
    #     'Fisher': {
    #         'cls': 0.8168,
    #         'kd': 0.6286
    #     },
    #     'Nwot': {
    #         'cls': 0.4029,
    #         'kd': 0.3187
    #     },
    #     'Snip': {
    #         'cls': 0.8466,
    #         'kd': 0.6722
    #     },
    #     'Vanilla': {
    #         'cls': 1,
    #         'kd': 0.8521
    #     }
    # }

    labels = result.keys()
    x = np.arange(len(labels))
    y1 = [v['cls'] for k, v in result.items()]
    y2 = [v['kd'] for k, v in result.items()]
    width = 0.3

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)

    rects1 = ax.bar(x - width / 2,
                    y1,
                    width,
                    label='Vanilla acc.',
                    color='salmon')
    rects2 = ax.bar(x + width / 2,
                    y2,
                    width,
                    label='Distill acc.',
                    color='skyblue')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Kendall's Tau", fontsize=GLOBAL_FONTSIZE, weight='bold')
    ax.set_xticks(x, labels, fontsize=GLOBAL_LEGENDSIZE - 2)
    yticks = np.arange(0.1, 1.2, 0.1)
    plt.tick_params(labelsize=GLOBAL_LEGENDSIZE - 2)
    ax.set_yticks(yticks, fontsize=GLOBAL_LABELSIZE)
    ax.set_ylim(0.2, 1.05)
    label_size = ax.legend().get_texts()
    [label.set_fontsize(GLOBAL_LEGENDSIZE - 3) for label in label_size]

    ax.bar_label(rects1,
                 padding=-26,
                 label_type='edge',
                 fontsize=GLOBAL_LEGENDSIZE - 2)
    ax.bar_label(rects2,
                 padding=-26,
                 label_type='edge',
                 fontsize=GLOBAL_LEGENDSIZE - 2)

    fig.tight_layout()
    plt.grid(linestyle='-.', alpha=0.9, lw=2)

    plt.savefig('./tmp/hist_rank_consistency.pdf',
                dpi=GLOBAL_DPI,
                bbox_inches='tight',
                pad_inches=PADINCHES,
                format='pdf')
    plt.clf()


def plot_param_cls_kd_ours_all():
    import json
    kd_dict = {}
    pre_dict = {}
    with open('./exps/diswot_sp_score_info.txt', 'r') as f:
        js = f.read()
        info_dict = json.loads(js)
        for k, v in info_dict.items():
            kd_dict[k] = v['gt']
            pre_dict[k] = v['pre']

    cls_dict = {}
    with open('./exps/s1-gt-cls.txt', 'r') as f:
        contents = f.readlines()
        for c in contents:
            k, v = c.split()
            cls_dict[k] = float(v)

    bar_width = 0.25

    merged_dict = {}
    for k, kd_acc in kd_dict.items():
        merged_dict[k] = [kd_acc, cls_dict[k], pre_dict[k]]

    labels = merged_dict.keys()
    x = np.arange(len(labels))
    kd_list = [v[0] for k, v in merged_dict.items()]
    cls_list = [v[1] for k, v in merged_dict.items()]
    pre_list = [0.001 - v[2] for k, v in merged_dict.items()]

    fig, ax1 = plt.subplots(1, 1, figsize=(30, 5))
    ax1.bar(x, cls_list, width=bar_width, label='Cls. ACC', fc='steelblue')
    ax1.bar(x + bar_width,
            kd_list,
            width=bar_width,
            label='KD. ACC',
            fc='seagreen')
    ax1.set_ylim([55, 80])

    ax2 = ax1.twinx()
    ax2.bar(x + bar_width * 2,
            pre_list,
            width=bar_width,
            label='DisWOT',
            fc='indianred')
    # ax2.set_ylim([])
    fig.legend(loc=1)
    ax1.set_xticks(x, labels)
    plt.savefig('./tmp/param_cls_kd_diswot.pdf',
                dpi=GLOBAL_DPI,
                bbox_inches='tight',
                pad_inches=PADINCHES,
                format='pdf')
    plt.clf()


def plot_param_cls_kd_ours():
    bar_width = 0.2

    labels = ['ResNet[713](259.89k)', 'ResNet[333](278.32k)']

    x = np.arange(len(labels))
    kd_list = [71.01, 70.76]
    cls_list = [69.13, 69.57]
    pre_list = [0.002 - 0.0015596, 0.002 - 0.0016668]
    pre_label = ['4.4e-4', '3.3e-4']

    fig, ax1 = plt.subplots(1, 1, figsize=FIGSIZE)

    ax1.spines['top'].set_color('black')
    ax1.spines['right'].set_color('black')
    ax1.spines['bottom'].set_color('black')
    ax1.spines['left'].set_color('black')
    ax1.spines['bottom'].set_linewidth(2)
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['top'].set_linewidth(2)
    ax1.spines['right'].set_linewidth(2)

    ax1.yaxis.label.set_size(GLOBAL_LEGENDSIZE)

    rects0 = ax1.bar(x,
                     cls_list,
                     width=bar_width,
                     label='Vanilla acc.',
                     fc='salmon')
    rects1 = ax1.bar(x + bar_width,
                     kd_list,
                     width=bar_width,
                     label='Distill acc.',
                     fc='skyblue')
    ax1.set_ylim([68.8, 71.5])
    ax1.set_ylabel('Top-1 Accuracy (%)',
                   fontsize=GLOBAL_FONTSIZE - 2,
                   weight='bold')

    for i, rect in enumerate(rects0):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2,
                 height - 0.18,
                 cls_list[i],
                 ha='center',
                 va='bottom',
                 fontsize=GLOBAL_LEGENDSIZE - 2)

    for i, rect in enumerate(rects1):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2,
                 height - 0.18,
                 kd_list[i],
                 ha='center',
                 va='bottom',
                 fontsize=GLOBAL_LEGENDSIZE - 2)

    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + bar_width * 2,
                     pre_list,
                     width=bar_width,
                     label='DisWOT Score',
                     fc='plum')
    ax2.set_ylim([0.0003, 0.0005])
    ax2.set_ylabel('DisWOT Score', fontsize=GLOBAL_FONTSIZE - 2, weight='bold')

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax2.yaxis.set_major_formatter(formatter)

    for i, rect in enumerate(rects2):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2,
                 height - 0.000014,
                 pre_label[i],
                 ha='center',
                 va='bottom',
                 fontsize=GLOBAL_LEGENDSIZE - 2)

    label_size = fig.legend(loc='upper right',
                            bbox_to_anchor=(0.83, 0.89)).get_texts()
    [label.set_fontsize(GLOBAL_LEGENDSIZE - 5) for label in label_size]

    ax1.set_xticks(x + bar_width, labels, fontsize=GLOBAL_LEGENDSIZE)
    ax1.tick_params(labelsize=GLOBAL_LEGENDSIZE - 3)
    ax2.tick_params(labelsize=GLOBAL_LEGENDSIZE - 3)
    plt.tight_layout()
    plt.grid(linestyle='-.', alpha=0.9, lw=2)
    plt.savefig('./tmp/param_cls_kd_diswot.pdf',
                dpi=GLOBAL_DPI,
                bbox_inches='tight',
                pad_inches=PADINCHES,
                format='pdf')
    plt.clf()


def plot_time_cost_vs_accuracy():
    # C100
    # RS, RL, BOHB, DARTS, GDAS, NWOT, TE-NAS, DisWOT, DisWOT+
    labels = [
        'RS', 'RL', 'BOHB', 'DARTS', 'GDAS', 'NWOT', 'TE-NAS', 'DisWOT',
        r'DisWOT($M_r$)'
    ]
    time_cost = [216000, 216000, 216000, 23000, 22000, 2200, 2200, 1200, 720]
    acc = [71.28, 71.71, 70.84, 66.24, 70.70, 73.31, 71.24, 74.21, 73.62]
    markers = ['X', ',', 'o', 'v', 'D', 'p', '>', '^', '*']
    x_offset = [-110000, -100000, -110000, 1000, 1000, 100, 100, 100, -160]
    y_offset = [-0.1, 0.2, -0.16, 0.2, 0.2, 0.2, 0.2, -0.2, -0.8]

    plt.figure(figsize=FIGSIZE)
    plt.rc('font', family='Times New Roman')
    plt.grid(linestyle='-.', alpha=0.9, lw=2)

    _len = len(labels)

    ax = plt.gca()
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)

    plt.annotate(
        '',
        xy=(1200, 73.62),
        xytext=(216000, 71.28),
        arrowprops=dict(arrowstyle='->',
                        lw=2),  #, facecolor='b', shrink=0.01, ec='b'),
        fontsize=GLOBAL_LEGENDSIZE,
        c='b')

    plt.text(22000, 72.5, '180x faster', fontsize=GLOBAL_LEGENDSIZE)

    for i in range(_len):
        plt.scatter(time_cost[i],
                    acc[i],
                    label=labels[i],
                    s=100,
                    marker=markers[i])
        plt.text(time_cost[i] + x_offset[i],
                 acc[i] + y_offset[i],
                 s=labels[i],
                 fontsize=GLOBAL_LEGENDSIZE - 5)

    plt.xscale('symlog')
    plt.ylim([64, 76])
    plt.tick_params(labelsize=GLOBAL_LEGENDSIZE)
    plt.xlabel('Log Time Cost (s)', fontsize=GLOBAL_LEGENDSIZE, weight='bold')
    plt.ylabel('Top-1 Accuracy (%)', fontsize=GLOBAL_FONTSIZE, weight='bold')
    plt.tight_layout()
    plt.legend(loc='lower left', fontsize=GLOBAL_LEGENDSIZE - 8)
    plt.savefig('./tmp/plot_time_cost_vs_accuracy.pdf',
                dpi=GLOBAL_DPI,
                bbox_inches='tight',
                pad_inches=PADINCHES,
                format='pdf')
    plt.clf()


if __name__ == '__main__':
    # plot_standalone_model_rank(xlabel='Ranking at ground-truth setting',
    #                            ylabel='Ranking by DisWOT',
    #                            foldername='./tmp',
    #                            file_name='standalone_ranks.pdf')
    # plot_ranking_stability(xlabel='Datasets',
    #                        ylabel='Ranking Correlation',
    #                        foldername='./tmp',
    #                        file_name='ranking_stability.pdf')

    plot_evolution_search_process()  # fig6
    plot_kd_box(foldername='./tmp', file_name='kd_box.pdf')  # fig5
    plot_different_teacher_diswot()  # fig3
    plot_hist_rank_consistency()  # fig1
    plot_param_cls_kd_ours()  # fig2
    plot_time_cost_vs_accuracy()  # fig4

    # plot_kd_zc_box(xlabel='Datasets',
    #                ylabel='Ranking Correlation',
    #                foldername='./tmp',
    #                file_name='kd_zc_box.pdf')
