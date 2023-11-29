from nas_201_api import NASBench201API

nb201_api = NASBench201API(
    file_path_or_dict='data/NAS-Bench-201-v1_0-e61699.pth', verbose=False)

candidate_list = [
    '|nor_conv_1x1~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|none~1|nor_conv_3x3~2|',
]

for arch in candidate_list:
    results = nb201_api.query_info_str_by_arch(arch, hp='200')
    print(results)
