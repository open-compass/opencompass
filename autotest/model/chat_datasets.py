from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.aime2025.aime2025_cascade_eval_gen_5e9f4f import \
        aime2025_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.gsm8k.gsm8k_gen import \
        gsm8k_datasets  # noqa: F401, E501
    # from opencompass.configs.datasets.mmlu_pro.mmlu_pro_gen import \
    #     mmlu_pro_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.HLE.hle_gen import \
        hle_datasets  # noqa: F401, E501
    # from opencompass.configs.datasets.humaneval.humaneval_gen import \
    #    humaneval_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.IFEval.IFEval_gen import \
        ifeval_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.infinitebench.infinitebenchretrievepasskey.infinitebench_retrievepasskey_gen import \
        InfiniteBench_retrievepasskey_datasets  # noqa: F401, E501

# humaneval_datasets = [humaneval_datasets[0]]
ifeval_datasets = [ifeval_datasets[0]]
# mmlu_pro_datasets = [mmlu_pro_datasets[0]]
hle_datasets = [hle_datasets[0]]
aime2025_datasets = [aime2025_datasets[0]]
aime2025_datasets[0]['n'] = 2

datasets = sum([v for k, v in locals().items() if k.endswith('_datasets')], [])

for d in datasets:
    d['reader_cfg']['test_range'] = '[0:4]'
    if 'dataset_cfg' in d['eval_cfg']['evaluator'] and 'reader_cfg' in d[
            'eval_cfg']['evaluator']['dataset_cfg']:
        d['eval_cfg']['evaluator']['dataset_cfg']['reader_cfg'][
            'test_range'] = '[0:4]'
    if 'llm_evaluator' in d['eval_cfg']['evaluator'] and 'dataset_cfg' in d[
            'eval_cfg']['evaluator']['llm_evaluator']:
        d['eval_cfg']['evaluator']['llm_evaluator']['dataset_cfg'][
            'reader_cfg']['test_range'] = '[0:4]'
