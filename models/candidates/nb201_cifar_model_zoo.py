import os
import sys

from models.nasbench201.utils import dict2config, get_cell_based_tiny_net

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def searched_nb201_cifar_diswot(num_classes=100):
    """DISWOT best student."""
    # searched_index = 10598
    searched_str = '|nor_conv_3x3~0|+|avg_pool_3x3~0|nor_conv_1x1~1|+|nor_conv_1x1~0|nor_conv_3x3~1|nor_conv_3x3~2|'
    arch_config = {
        'name': 'infer.tiny',
        'C': 16,
        'N': 5,
        'arch_str': searched_str,
        'num_classes': num_classes
    }
    net_config = dict2config(arch_config, None)
    return get_cell_based_tiny_net(net_config)


def searched_nb201_cifar_diswot_sp(num_classes=100):
    """DISWOT best student with only sp."""
    # searched_index = 2725 test acc=93.45%
    searched_str = '|nor_conv_1x1~0|+|nor_conv_3x3~0|none~1|+|nor_conv_3x3~0|skip_connect~1|nor_conv_3x3~2|'
    arch_config = {
        'name': 'infer.tiny',
        'C': 16,
        'N': 5,
        'arch_str': searched_str,
        'num_classes': num_classes
    }
    net_config = dict2config(arch_config, None)
    return get_cell_based_tiny_net(net_config)


def searched_nb201_cifar_rminas(num_classes=100):
    """RMINAS best student."""
    # searched_index = 10598
    searched_str = '|nor_conv_1x1~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|none~1|nor_conv_3x3~2|'
    arch_config = {
        'name': 'infer.tiny',
        'C': 16,
        'N': 5,
        'arch_str': searched_str,
        'num_classes': num_classes
    }
    net_config = dict2config(arch_config, None)
    return get_cell_based_tiny_net(net_config)


def searched_nb201_cifar_enas(num_classes=100):
    """ENAS best student."""
    # searched_index = 10598
    searched_str = '|skip_connect~0|+|avg_pool_3x3~0|skip_connect~1|+|avg_pool_3x3~0|skip_connect~1|skip_connect~2|'
    arch_config = {
        'name': 'infer.tiny',
        'C': 16,
        'N': 5,
        'arch_str': searched_str,
        'num_classes': num_classes
    }
    net_config = dict2config(arch_config, None)
    return get_cell_based_tiny_net(net_config)


def searched_nb201_cifar_gdas(num_classes=100):
    """GDAS best student."""
    # searched_index = 10598
    searched_str = '|nor_conv_3x3~0|+|nor_conv_3x3~0|none~1|+|nor_conv_3x3~0|nor_conv_3x3~1|nor_conv_3x3~2|'
    arch_config = {
        'name': 'infer.tiny',
        'C': 16,
        'N': 5,
        'arch_str': searched_str,
        'num_classes': num_classes
    }
    net_config = dict2config(arch_config, None)
    return get_cell_based_tiny_net(net_config)


def searched_nb201_cifar_darts(num_classes=100):
    """DARTS best student."""
    # searched_index = 10598
    searched_str = '|skip_connect~0|+|skip_connect~0|skip_connect~1|+|skip_connect~0|skip_connect~1|skip_connect~2|'
    arch_config = {
        'name': 'infer.tiny',
        'C': 16,
        'N': 5,
        'arch_str': searched_str,
        'num_classes': num_classes
    }
    net_config = dict2config(arch_config, None)
    return get_cell_based_tiny_net(net_config)


def searched_nb201_cifar_setn(num_classes=100):
    """SETN best student."""
    # searched_index = 10598
    searched_str = '|nor_conv_3x3~0|+|skip_connect~0|skip_connect~1|+|skip_connect~0|skip_connect~1|avg_pool_3x3~2|'
    arch_config = {
        'name': 'infer.tiny',
        'C': 16,
        'N': 5,
        'arch_str': searched_str,
        'num_classes': num_classes
    }
    net_config = dict2config(arch_config, None)
    return get_cell_based_tiny_net(net_config)


def searched_nb201_cifar_spos(num_classes=100):
    """SPOS best student."""
    # searched_index = 13688
    searched_str = '|skip_connect~0|+|nor_conv_1x1~0|nor_conv_3x3~1|+|nor_conv_1x1~0|avg_pool_3x3~1|nor_conv_3x3~2|'
    arch_config = {
        'name': 'infer.tiny',
        'C': 16,
        'N': 5,
        'arch_str': searched_str,
        'num_classes': num_classes
    }
    net_config = dict2config(arch_config, None)
    return get_cell_based_tiny_net(net_config)


def searched_nb201_cifar_ea(num_classes=100):
    """Evolution algorithm best student."""
    # searched_index = 10598
    searched_str = '|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2|'
    arch_config = {
        'name': 'infer.tiny',
        'C': 16,
        'N': 5,
        'arch_str': searched_str,
        'num_classes': num_classes
    }
    net_config = dict2config(arch_config, None)
    return get_cell_based_tiny_net(net_config)


def searched_nb201_cifar_random(num_classes=100):
    """Random Baseline best student."""
    # searched_index = 10598
    searched_str = '|skip_connect~0|+|nor_conv_3x3~0|skip_connect~1|+|nor_conv_3x3~0|nor_conv_1x1~1|avg_pool_3x3~2|'
    arch_config = {
        'name': 'infer.tiny',
        'C': 16,
        'N': 5,
        'arch_str': searched_str,
        'num_classes': num_classes
    }
    net_config = dict2config(arch_config, None)
    return get_cell_based_tiny_net(net_config)


def searched_nb201_cifar_reinforce(num_classes=100):
    """REINFORCE best student."""
    # searched_index = 10598
    searched_str = '|nor_conv_3x3~0|+|nor_conv_1x1~0|nor_conv_1x1~1|+|nor_conv_3x3~0|nor_conv_3x3~1|nor_conv_3x3~2|'
    arch_config = {
        'name': 'infer.tiny',
        'C': 16,
        'N': 5,
        'arch_str': searched_str,
        'num_classes': num_classes
    }
    net_config = dict2config(arch_config, None)
    return get_cell_based_tiny_net(net_config)


def searched_nb201_cifar_rnasd(num_classes=100):
    """RNASD best student."""
    # searched_index = 10598
    searched_str = '|nor_conv_3x3~0|+|nor_conv_1x1~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_1x1~1|nor_conv_3x3~2|'
    arch_config = {
        'name': 'infer.tiny',
        'C': 16,
        'N': 5,
        'arch_str': searched_str,
        'num_classes': num_classes
    }
    net_config = dict2config(arch_config, None)
    return get_cell_based_tiny_net(net_config)


def searched_nb201_cifar_rdnas(num_classes=100):
    """RDNAS best student."""
    # searched_index = 10598
    searched_str = '|nor_conv_3x3~0|+|nor_conv_3x3~0|avg_pool_3x3~1|+|skip_connect~0|avg_pool_3x3~1|nor_conv_3x3~2|'
    arch_config = {
        'name': 'infer.tiny',
        'C': 16,
        'N': 5,
        'arch_str': searched_str,
        'num_classes': num_classes
    }
    net_config = dict2config(arch_config, None)
    return get_cell_based_tiny_net(net_config)


def searched_nb201_cifar_zen(num_classes=100):
    """ZenNAS best student."""
    searched_str = '|skip_connect~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|skip_connect~1|nor_conv_3x3~2|'
    arch_config = {
        'name': 'infer.tiny',
        'C': 16,
        'N': 5,
        'arch_str': searched_str,
        'num_classes': num_classes
    }
    net_config = dict2config(arch_config, None)
    return get_cell_based_tiny_net(net_config)


def searched_nb201_cifar_nwot(num_classes=100):
    """NWOT best student."""
    searched_str = '|nor_conv_1x1~0|+|nor_conv_3x3~0|nor_conv_1x1~1|+|nor_conv_1x1~0|nor_conv_3x3~1|nor_conv_1x1~2|'
    arch_config = {
        'name': 'infer.tiny',
        'C': 16,
        'N': 5,
        'arch_str': searched_str,
        'num_classes': num_classes
    }
    net_config = dict2config(arch_config, None)
    return get_cell_based_tiny_net(net_config)


def searched_nb201_cifar_synflow(num_classes=100):
    """synflow best student."""
    searched_str = '|nor_conv_3x3~0|+|avg_pool_3x3~0|nor_conv_1x1~1|+|none~0|skip_connect~1|nor_conv_3x3~2|'
    arch_config = {
        'name': 'infer.tiny',
        'C': 16,
        'N': 5,
        'arch_str': searched_str,
        'num_classes': num_classes
    }
    net_config = dict2config(arch_config, None)
    return get_cell_based_tiny_net(net_config)


def searched_nb201_cifar_diswot_ss(num_classes=100):
    searched_str = '|skip_connect~0|+|nor_conv_3x3~0|nor_conv_1x1~1|+|nor_conv_1x1~0|nor_conv_3x3~1|nor_conv_3x3~2|'
    arch_config = {
        'name': 'infer.tiny',
        'C': 16,
        'N': 5,
        'arch_str': searched_str,
        'num_classes': num_classes
    }
    net_config = dict2config(arch_config, None)
    return get_cell_based_tiny_net(net_config)
