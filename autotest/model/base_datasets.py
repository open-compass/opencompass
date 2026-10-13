from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.gpqa.gpqa_few_shot_ppl_4b5a83 import \
        gpqa_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_17d0dc import \
        gsm8k_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.infinitebench.infinitebenchretrievepasskey.infinitebench_retrievepasskey_gen import \
        InfiniteBench_retrievepasskey_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_few_shot_gen_bfaf90 import \
        mmlu_pro_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.winogrande.winogrande_5shot_ll_252f01 import \
        winogrande_datasets  # noqa: F401, E501

# humaneval_datasets = [humaneval_datasets[0]]
mmlu_pro_datasets = [mmlu_pro_datasets[0]]

datasets = sum([v for k, v in locals().items() if k.endswith('_datasets')], [])

for d in datasets:
    d['reader_cfg']['test_range'] = '[0:4]'
