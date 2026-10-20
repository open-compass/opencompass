from mmengine import read_base

with read_base():
    from opencompass.configs.datasets.korbench.korbench_mixed_gen_d00bdd import \
        korbench_mixed_datasets as mixed_datasets
    from opencompass.configs.datasets.korbench.korbench_single_0_shot_gen import \
        korbench_0shot_single_datasets as zero_shot_datasets
    from opencompass.configs.datasets.korbench.korbench_single_3_shot_gen import \
        korbench_3shot_single_datasets as three_shot_datasets
    from opencompass.configs.models.hf_internlm.hf_internlm2_5_7b import \
        models as hf_internlm2_5_7b

datasets = zero_shot_datasets + three_shot_datasets + mixed_datasets
models = hf_internlm2_5_7b
