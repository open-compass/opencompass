from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.babilong.babilong_256k_gen import \
        babiLong_256k_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.longbench.longbench import \
        longbench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.needlebench.needlebench_128k.needlebench_128k import \
        needlebench_datasets as needlebench_128k_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.ruler.ruler_128k_gen import \
        ruler_datasets as ruler_128k_datasets  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_5_7b_chat_1m import \
        models as lmdeploy_internlm2_5_7b_chat_1m_model  # noqa: F401, E501
    # Summary Groups
    from opencompass.configs.summarizers.groups.babilong import \
        babilong_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.longbench import \
        longbench_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.ruler import \
        ruler_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.needlebench import \
        needlebench_128k_summarizer  # noqa: F401, E501

    from ...volc import infer  # noqa: F401, E501

models = sum([v for k, v in locals().items() if k.endswith('_model')], [])
datasets = [
    v[0] for k, v in locals().items()
    if k.endswith('_datasets') and isinstance(v, list) and len(v) > 0
]

for d in datasets:
    d['reader_cfg']['test_range'] = '[0:16]'
