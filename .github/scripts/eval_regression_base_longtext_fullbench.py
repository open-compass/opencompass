from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.longbench.longbench import \
        longbench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.needlebench.needlebench_base.needlebench_base_gen import \
        needlebench_datasets  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_5_7b_chat_1m import \
        models as lmdeploy_internlm2_5_7b_chat_1m_model  # noqa: F401, E501
    # summarizer
    from opencompass.configs.summarizers.groups.longbench import \
        longbench_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.needlebench import \
        needlebench_internal_200k_summarizer  # noqa: F401, E501
    from opencompass.configs.summarizers.needlebench import (
        needlebench_internal_32k_summarizer,
        needlebench_internal_100k_summarizer)

    from ...volc import infer  # noqa: F401, E501

needlebench_internal_32k_summary_groups = needlebench_internal_32k_summarizer[
    'summary_groups']
needlebench_internal_100k_summary_groups = (
    needlebench_internal_100k_summarizer['summary_groups'])
needlebench_internal_200k_summary_groups = (
    needlebench_internal_200k_summarizer['summary_groups'])

models = sum([v for k, v in locals().items() if k.endswith('_model')], [])
datasets = [
    v[0] for k, v in locals().items()
    if k.endswith('_datasets') and isinstance(v, list) and len(v) > 0
]

for d in datasets:
    d['reader_cfg']['test_range'] = '[0:16]'
