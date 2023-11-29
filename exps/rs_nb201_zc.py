import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from nas_201_api import NASBench201API
from rank_consisteny import kendalltau, pearson, spearman
from torch import Tensor

from dataset.cifar10 import get_cifar10_dataloaders
from models.nasbench201.utils import dict2config, get_cell_based_tiny_net

nb201_api = NASBench201API(
    file_path_or_dict='data/NAS-Bench-201-v1_0-e61699.pth', verbose=False)

# # two Inputs


def elemenet_wise_sum(A, B):
    return A + B


def element_wise_difference(A, B):
    return A - B


def element_wise_product(A, B):
    return A * B


def matrix_multiplication(A, B):
    return A @ B


def lesser_than(A, B) -> bool:
    return (A < B).bool()


def greater_than(A, B) -> bool:
    return (A > B).bool()


def equal_to(A, B) -> bool:
    return (A == B).bool()


def hamming_distance(A, B) -> Tensor:
    A = torch.heaviside(A, values=[0])
    B = torch.heaviside(B, values=[0])
    return sum(A != B)


def kl_divergence(A, B) -> Tensor:
    return torch.nn.KLDivLoss('batchmean')(A, B)


def cosine_similarity(A, B) -> Tensor:
    A = A.reshape(A.shape[0], -1)
    B = A.reshape(B.shape[0], -1)
    C = torch.nn.CosineSimilarity()(A, B)
    return torch.sum(C)


# # One Input
def element_wise_log(A) -> Tensor:
    A[A <= 0] == 1
    return torch.log(A)


def element_wise_abslog(A) -> Tensor:
    A[A == 0] = 1
    A = torch.abs(A)
    return torch.log(A)


def element_wise_abs(A) -> Tensor:
    return torch.log(A)


def element_wise_pow(A) -> Tensor:
    return torch.pow(A, 2)


def element_wise_exp(A) -> Tensor:
    return torch.exp(A)


def normalize(A) -> Tensor:
    m = torch.mean(A)
    s = torch.std(A)
    C = (A - m) / s
    C[C != C] = 0
    return C


def element_wise_relu(A) -> Tensor:
    return F.relu(A)


def element_wise_sign(A) -> Tensor:
    return torch.sign(A)


def element_wise_invert(A) -> Tensor:
    return 1 / A


def frobenius_norm(A) -> Tensor:
    return torch.norm(A, p='fro')


def element_wise_normalized_sum(A) -> Tensor:
    return torch.sum(A) / A.numel()


def l1_norm(A) -> Tensor:
    return torch.sum(torch.abs(A)) / A.numel()


def softmax(A) -> Tensor:
    return F.softmax(A)


def sigmoid(A) -> Tensor:
    return F.sigmoid(A)


def binary_operation(A, B, idx=None):
    # 10
    binary_keys = [
        'elemenet_wise_sum', 'element_wise_difference', 'element_wise_product',
        'lesser_than', 'greater_than', 'equal_to', 'hamming_distance',
        'kl_divergence', 'cosine_similarity', 'matrix_multiplication'
    ]
    if idx is None:
        idx = random.choice(range(len(binary_keys)))

    assert idx < len(binary_keys)

    binaries = {
        'elemenet_wise_sum': elemenet_wise_sum,
        'element_wise_difference': element_wise_difference,
        'element_wise_product': element_wise_product,
        'matrix_multiplication': matrix_multiplication,
        'lesser_than': lesser_than,
        'greater_than': greater_than,
        'equal_to': equal_to,
        'hamming_distance': hamming_distance,
        'kl_divergence': kl_divergence,
        'cosine_similarity': cosine_similarity
    }
    return binaries[binary_keys[idx]](A, B)


def unary_operation(A, idx=None):
    # 14
    unary_keys = [
        'element_wise_log', 'element_wise_abslog', 'element_wise_abs',
        'element_wise_pow', 'element_wise_exp', 'normalize',
        'element_wise_relu', 'element_wise_sign', 'element_wise_invert',
        'frobenius_norm', 'element_wise_normalized_sum', 'l1_norm', 'softmax',
        'sigmoid'
    ]
    if idx is None:
        idx = random.choice(range(len(unary_keys)))

    assert idx < len(unary_keys)

    unaries = {
        'element_wise_log': element_wise_log,
        'element_wise_abslog': element_wise_abslog,
        'element_wise_abs': element_wise_abs,
        'matrix_multiplication': matrix_multiplication,
        'element_wise_pow': element_wise_pow,
        'element_wise_exp': element_wise_exp,
        'normalize': normalize,
        'element_wise_relu': element_wise_relu,
        'element_wise_sign': element_wise_sign,
        'element_wise_invert': element_wise_invert,
        'frobenius_norm': frobenius_norm,
        'element_wise_normalized_sum': element_wise_normalized_sum,
        'l1_norm': l1_norm,
        'softmax': softmax,
        'sigmoid': sigmoid,
    }
    return unaries[unary_keys[idx]](A)


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
        'num_classes': 10
    }
    net_config = dict2config(arch_config, None)
    model = get_cell_based_tiny_net(net_config)
    xinfo = nb201_api.get_more_info(choiced_index, dataset='cifar10', hp='200')
    return choiced_index, model, xinfo['test-accuracy']


def generate_random_zc() -> list:
    # for A: 14
    idx1 = random.choice(range(14))
    # for B: 14
    idx2 = random.choice(range(14))
    # for A,B: 10
    idx3 = random.choice(range(10))

    return idx1, idx2, idx3


def execute_zc_workflow(model, zc_idxs):
    """Random Generate ZC

    Tree Structure:
        Generate Input(select two of them):
            - feature[-2].
            - grad info.
        Generate Output(A, B):
            - for A: unary
            - for B: unary
            - for A, B: binary
                - if not value, continue binary
    """
    train_loader, val_loader = get_cifar10_dataloaders(batch_size=4,
                                                       num_workers=1)
    dataiter = iter(train_loader)
    try:
        img, label = next(dataiter)
    except StopIteration as e:
        dataiter = iter(dataloader)
        img, label = next(dataiter)

    tout1, tlogits1 = model.forward_with_features(img)
    input = torch.randn(size=list(img.shape))
    tout2, tlogits2 = model.forward_with_features(input)

    input1 = tout1[-2]
    input2 = tout2[-2]

    try:
        print(f' =>input {input1.shape} / {input2.shape}')
        A = unary_operation(input1, zc_idxs[0])
        B = unary_operation(input2, zc_idxs[1])
        print(f' =>AB {A.shape} / {B.shape}')
        if len(list(A.shape)) <= 1:
            return A.item()
        if len(list(B.shape)) <= 1:
            return B.item()

        C = binary_operation(A, B, zc_idxs[2])
        print(C.shape, type(C))
        if len(list(C.shape)) > 1:
            C = element_wise_normalized_sum(C)
        return C.item()
    except BaseException as e:
        print(f'error from {e}')
    return -1


def ranking_consistency_of_zc(num_sample=10):
    zc_idxs = generate_random_zc()

    gt_score = []
    zc_score = []
    for i in range(num_sample):
        idx, model, gt = random_sample_and_get_gt()
        zc = execute_zc_workflow(model, zc_idxs)
        print(f'the {i}-th zc result is : {zc}')
        if zc != -1:
            zc_score.append(zc)
            gt_score.append(gt)
    return kendalltau(gt_score, zc_score)


if __name__ == '__main__':
    kd_list = []
    for _ in range(20):
        kd = ranking_consistency_of_zc()
        kd_list.append(kd)

    print(f"best kendall's tau: {max(kd_list)} All kendall's tau: {kd_list}")
