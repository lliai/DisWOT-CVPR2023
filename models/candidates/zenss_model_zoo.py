import models
from models.candidates.mutable import MasterNet


def zen_cifar_res32_1M(num_classes=100):
    searched_structure = 'SuperConvK3BNRELU(3,88,1,1)SuperResK1K7K1(88,120,1,16,1)SuperResK1K7K1(120,192,2,16,3)SuperResK1K5K1(192,224,1,24,4)SuperResK1K5K1(224,96,2,24,2)SuperResK1K3K1(96,168,2,40,3)SuperResK1K3K1(168,112,1,48,3)SuperConvK1BNRELU(112,512,1,1)'
    net = MasterNet(plainnet_struct=searched_structure,
                    num_classes=num_classes)
    return net


def zen_cifar_res32_05M(num_classes=100):
    searched_structure = 'SuperConvK3BNRELU(3,64,1,1)SuperResK1K5K1(64,168,1,16,3)SuperResK1K3K1(168,80,2,32,4)SuperResK1K5K1(80,112,2,16,3)SuperResK1K5K1(112,144,1,24,3)SuperResK1K3K1(144,32,2,40,1)SuperConvK1BNRELU(32,512,1,1)'
    net = MasterNet(plainnet_struct=searched_structure,
                    num_classes=num_classes)
    return net


def zen_cifar_res32_2M(num_classes=100):
    searched_structure = 'SuperConvK3BNRELU(3,32,1,1)SuperResK1K5K1(32,120,1,40,1)SuperResK1K5K1(120,176,2,32,3)SuperResK1K7K1(176,272,1,24,3)SuperResK1K3K1(272,176,1,56,3)SuperResK1K3K1(176,176,1,64,4)SuperResK1K5K1(176,216,2,40,2)SuperResK1K3K1(216,72,2,56,2)SuperConvK1BNRELU(72,512,1,1)'
    net = MasterNet(plainnet_struct=searched_structure,
                    num_classes=num_classes)
    return net


def diswot_imagenet_res18_7M(num_classes=1000):
    searched_structure = 'SuperConvK3BNRELU(3,256,2,1)SuperResK5K5(256,24,2,16,1)SuperResK1K7K1(24,720,2,80,1)SuperResK3K3(720,192,2,16,1)SuperResK7K7(192,328,2,64,3)SuperConvK1BNRELU(328,208,1,1)'
    net = MasterNet(plainnet_struct=searched_structure,
                    num_classes=num_classes)
    return net


def diswot_imagenet_res18_13M(num_classes=1000):
    searched_structure = 'SuperConvK3BNRELU(3,48,2,1)SuperResK5K5(48,48,2,80,1)SuperResK5K5(48,216,2,24,1)SuperResK3K3(216,496,2,128,3)SuperResK7K7(496,256,2,80,3)SuperConvK1BNRELU(256,2048,1,1)'
    net = MasterNet(plainnet_struct=searched_structure,
                    num_classes=num_classes)
    return net


def diswot_imagenet_res18_11M(num_classes=1000):
    searched_structure = 'SuperConvK7BNRELU(3,96,2,1)SuperResK3K3(96,64,2,64,2)SuperResK3K3(64,128,2,128,2)SuperResK3K3(128,256,2,256,2)SuperResK3K3(256,512,2,512,2)'
    net = MasterNet(plainnet_struct=searched_structure,
                    num_classes=num_classes)
    return net


# @MODELS.register_module()
def diswot_imagenet_res18_8M(num_classes=1000):
    searched_structure = 'SuperConvK7BNRELU(3,64,2,1)SuperResK3K3(64,64,2,64,2)SuperResK3K3(64,128,2,128,2)SuperResK3K3(128,256,2,256,2)SuperResK5K5(256,344,2,344,1)'
    net = MasterNet(plainnet_struct=searched_structure,
                    num_classes=num_classes)
    return net


# @MODELS.register_module()
def diswot_imagenet_res18_11_6M(num_classes=1000):
    searched_structure = 'SuperConvK7BNRELU(3,64,2,1)SuperResK3K3(64,64,2,64,2)SuperResK3K3(64,128,2,128,2)SuperResK3K3(128,256,2,256,2)SuperResK3K3(256,512,2,512,2)'
    net = MasterNet(plainnet_struct=searched_structure,
                    num_classes=num_classes)
    return net
