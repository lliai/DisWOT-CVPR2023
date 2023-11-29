import random

import torch
import torch.nn.functional as F
from mmcv.cnn.utils import get_model_complexity_info

import models  # noqa: F401,F403
from dataset.cifar100 import get_cifar100_dataloaders
from models.candidates.mutable import MasterNet
from models.candidates.mutable.searchspace.search_space_xxbl import \
    gen_search_space
from models.candidates.mutable.utils import (PlainNet,
                                             create_netblock_list_from_str)
from predictor.pruners import predictive


def get_new_random_structure_str(the_net,
                                 get_search_space_func,
                                 num_replaces=1):
    """
    what is meaning of replaces?
    """
    assert isinstance(the_net, PlainNet)
    selected_random_id_set = set()
    for replace_count in range(num_replaces):
        random_id = random.randint(0, len(the_net.block_list) - 1)
        if random_id in selected_random_id_set:
            continue
        selected_random_id_set.add(random_id)
        to_search_student_blocks_list_list = get_search_space_func(
            the_net.block_list, random_id)

        to_search_student_blocks_list = [
            x for sublist in to_search_student_blocks_list_list
            for x in sublist
        ]
        new_student_block_str = random.choice(to_search_student_blocks_list)

        if len(new_student_block_str) > 0:
            new_student_block = create_netblock_list_from_str(
                new_student_block_str, no_create=True)
            assert len(new_student_block) == 1
            new_student_block = new_student_block[0]
            if random_id > 0:
                last_block_out_channels = the_net.block_list[random_id -
                                                             1].out_channels
                new_student_block.set_in_channels(last_block_out_channels)
            the_net.block_list[random_id] = new_student_block
        else:
            # replace with empty block
            the_net.block_list[random_id] = None
    pass  # end for

    # adjust channels and remove empty layer
    tmp_new_block_list = [x for x in the_net.block_list if x is not None]
    last_channels = the_net.block_list[0].out_channels
    for block in tmp_new_block_list[1:]:
        block.set_in_channels(last_channels)
        last_channels = block.out_channels
    the_net.block_list = tmp_new_block_list

    new_random_structure_str = the_net.split(split_layer_threshold=6)
    return new_random_structure_str


if __name__ == '__main__':
    plainnet_struct = 'SuperConvK3BNRELU(3,64,1,1)SuperResK1K5K1(64,168,1,16,3)SuperResK1K3K1(168,80,2,32,4)SuperResK1K5K1(80,112,2,16,3)SuperResK1K5K1(112,144,1,24,3)SuperResK1K3K1(144,32,2,40,1)SuperConvK1BNRELU(32,512,1,1)'

    dataload_info = ['random', 3, 100]

    # print(pretty_format(plainnet_struct))
    net = MasterNet(plainnet_struct=plainnet_struct)
    init_structure_str = str(net)

    _, dataloader, n_data = get_cifar100_dataloaders(batch_size=64,
                                                     num_workers=2,
                                                     is_instance=True)

    score = predictive.find_measures(net,
                                     dataloader,
                                     dataload_info=dataload_info,
                                     measure_names=['zen'],
                                     loss_fn=F.cross_entropy,
                                     device=torch.device('cpu'))

    print(score)

    candidates_list = []

    best_score = 0
    best_candidate = None
    best_flops = None
    best_params = None

    for i in range(300):
        new_structures_str = get_new_random_structure_str(
            net,
            get_search_space_func=gen_search_space,
            num_replaces=2,
        )
        candidates_list.append(new_structures_str)

        new_net = MasterNet(plainnet_struct=new_structures_str)

        score = predictive.find_measures(new_net,
                                         dataloader,
                                         dataload_info=dataload_info,
                                         measure_names=['zen'],
                                         loss_fn=F.cross_entropy,
                                         device=torch.device('cpu'))

        flops, params = get_model_complexity_info(new_net,
                                                  input_shape=(3, 32, 32),
                                                  print_per_layer_stat=False)

        print(new_structures_str, score, flops, params)
        if score > best_score:
            best_score = score
            best_candidate = new_structures_str
            best_flops = flops
            best_params = params

        del new_net

    print('Finally Desicion: ', best_candidate, best_score, best_flops,
          best_params)
