from mmengine.config import read_base

from opencompass.models import TurboMindModel

with read_base():
    from opencompass.configs.datasets.longbench.longbench import \
        longbench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.needlebench.needlebench_base.needlebench_base_gen import \
        needlebench_datasets  # noqa: F401, E501
    # summarizer
    from opencompass.configs.summarizers.groups.longbench import \
        longbench_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.needlebench import \
        needlebench_internal_200k_summarizer  # noqa: F401, E501
    from opencompass.configs.summarizers.needlebench import (
        needlebench_internal_32k_summarizer,
        needlebench_internal_100k_summarizer)

    from ...rjob import eval, infer  # noqa: F401, E501

needlebench_internal_32k_summary_groups = needlebench_internal_32k_summarizer[
    'summary_groups']
needlebench_internal_100k_summary_groups = (
    needlebench_internal_100k_summarizer['summary_groups'])
needlebench_internal_200k_summary_groups = (
    needlebench_internal_200k_summarizer['summary_groups'])

models = [
    dict(
        type=TurboMindModel,
        abbr='qwen3-8b-base-turbomind',
        path='Qwen/Qwen3-8B-Base',
        engine_config=dict(session_len=264192, max_batch_size=8, tp=1),
        gen_config=dict(top_k=1,
                        temperature=1e-6,
                        top_p=0.9,
                        max_new_tokens=2048,
                        min_out_len=2),
        max_seq_len=264192,
        max_out_len=500,
        batch_size=1,
        drop_middle=True,
        run_cfg=dict(num_gpus=1),
    )
]

datasets = [
    v[0] for k, v in locals().items()
    if k.endswith('_datasets') and isinstance(v, list) and len(v) > 0
]

for d in datasets:
    d['reader_cfg']['test_range'] = '[0:16]'
