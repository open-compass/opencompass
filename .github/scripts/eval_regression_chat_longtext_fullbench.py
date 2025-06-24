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
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_5_7b import \
        models as lmdeploy_internlm2_5_7b_model  # noqa: F401, E501
    # Summary Groups
    from opencompass.configs.summarizers.groups.babilong import \
        babilong_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.longbench import \
        longbench_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.ruler import \
        ruler_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.needlebench import \
        needlebench_128k_summarizer  # noqa: F401, E501

    from ...volc import infer as volc_infer  # noqa: F401, E501

needlebench_128k_summary_groups = needlebench_128k_summarizer['summary_groups']
summary_groups = sum(
    [v for k, v in locals().items() if k.endswith('_summary_groups')], [])

summarizer = dict(
    dataset_abbrs=[
        ['ruler_8k', 'naive_average'],
        ['ruler_32k', 'naive_average'],
        ['ruler_128k', 'naive_average'],
        ['NeedleBench-Overall-Score-8K', 'weighted_average'],
        ['NeedleBench-Overall-Score-32K', 'weighted_average'],
        ['NeedleBench-Overall-Score-128K', 'weighted_average'],
        ['longbench', 'naive_average'],
        ['longbench_zh', 'naive_average'],
        ['longbench_en', 'naive_average'],
        ['babilong_0k', 'naive_average'],
        ['babilong_4k', 'naive_average'],
        ['babilong_16k', 'naive_average'],
        ['babilong_32k', 'naive_average'],
        ['babilong_128k', 'naive_average'],
        ['babilong_256k', 'naive_average'],
        '',
        'longbench_single-document-qa',
        'longbench_multi-document-qa',
        'longbench_summarization',
        'longbench_few-shot-learning',
        'longbench_synthetic-tasks',
        'longbench_code-completion',
    ],
    summary_groups=summary_groups,
)

models = sum([v for k, v in locals().items() if k.endswith('_model')], [])
datasets = [
    v[0] for k, v in locals().items()
    if k.endswith('_datasets') and isinstance(v, list) and len(v) > 0
]

for d in datasets:
    d['reader_cfg']['test_range'] = '[0:16]'

for m in models:
    m['abbr'] = m['abbr'] + '_fullbench'
    if 'turbomind' in m['abbr'] or 'lmdeploy' in m['abbr']:
        m['engine_config']['max_batch_size'] = 1
        m['batch_size'] = 1
        m['engine_config']['tp'] = 4
