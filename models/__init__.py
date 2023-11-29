from .candidates import *  # noqa: F401,F403
from .mobilenetv2 import mobile_half
from .resnet import (resnet8, resnet8x4, resnet14, resnet20, resnet32,
                     resnet32x4, resnet44, resnet56, resnet110, resnet110_zen)
from .resnetv2 import ResNet50
from .ShuffleNetv1 import ShuffleV1
from .ShuffleNetv2 import ShuffleV2
from .vgg import vgg8_bn, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2

model_dict = {
    'resnet8': resnet8,
    'resnet14': resnet14,
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet56': resnet56,
    'resnet110': resnet110,
    'resnet8x4': resnet8x4,
    'resnet32x4': resnet32x4,
    'ResNet50': ResNet50,
    'wrn_16_1': wrn_16_1,
    'wrn_16_2': wrn_16_2,
    'wrn_40_1': wrn_40_1,
    'wrn_40_2': wrn_40_2,
    'vgg8': vgg8_bn,
    'vgg11': vgg11_bn,
    'vgg13': vgg13_bn,
    'vgg16': vgg16_bn,
    'vgg19': vgg19_bn,
    'MobileNetV2': mobile_half,
    'ShuffleV1': ShuffleV1,
    'ShuffleV2': ShuffleV2,
    'zen_cifar_res32_1M': zen_cifar_res32_1M,
    'zen_cifar_res32_05M': zen_cifar_res32_05M,
    'zen_cifar_res32_2M': zen_cifar_res32_2M,
    'diswot_imagenet_res18_7M': diswot_imagenet_res18_7M,
    'searched_nb201_cifar_diswot': searched_nb201_cifar_diswot,
    'searched_nb201_cifar_diswot_sp': searched_nb201_cifar_diswot_sp,
    'searched_nb201_cifar_rminas': searched_nb201_cifar_rminas,
    'searched_nb201_cifar_enas': searched_nb201_cifar_enas,
    'searched_nb201_cifar_gdas': searched_nb201_cifar_gdas,
    'searched_nb201_cifar_darts': searched_nb201_cifar_darts,
    'searched_nb201_cifar_setn': searched_nb201_cifar_setn,
    'searched_nb201_cifar_ea': searched_nb201_cifar_ea,
    'searched_nb201_cifar_random': searched_nb201_cifar_random,
    'searched_nb201_cifar_reinforce': searched_nb201_cifar_reinforce,
    'searched_nb201_cifar_rnasd': searched_nb201_cifar_rnasd,
    'searched_nb201_cifar_rdnas': searched_nb201_cifar_rdnas,
    'searched_nb201_cifar_spos': searched_nb201_cifar_spos,
    'searched_nb201_cifar_nwot': searched_nb201_cifar_nwot,
    'searched_nb201_cifar_zen': searched_nb201_cifar_zen,
    'searched_nb201_cifar_diswot_ss': searched_nb201_cifar_diswot_ss,
}
