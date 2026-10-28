from mmengine.config import read_base

with read_base():
    from autotest.all.config import models  # noqa: F401
    from autotest.all.config import MOCK_API_MAX_WORKERS
    from autotest.all.config import common_eval as eval  # noqa: F401
    from autotest.all.config import concurrent_infer as infer  # noqa: F401
    from autotest.all.config import set_mock_test_range
    from opencompass.configs.datasets.aa_lcr.aa_lcr_gen import aa_lcr_datasets
    from opencompass.configs.datasets.babilong.babilong_0k_rawprompt_gen import \
        babiLong_0k_datasets
    from opencompass.configs.datasets.longbench.longbench_rawprompt import \
        longbench_rawprompt_datasets as LongBenchv1_datasets
    # Datasets
    from opencompass.configs.datasets.longbenchv2.longbenchv2_rawprompt_gen_0903e0 import \
        LongBenchv2_datasets as LongBenchv2_datasets
    from opencompass.configs.datasets.mrcr.mrcr_v2_full_rawprompt_gen import \
        mrcr_datasets
    from opencompass.configs.datasets.needlebench.needlebench_8k.needlebench_8k_rawprompt import \
        needlebench_datasets as needlebench_8k_datasets
    from opencompass.configs.datasets.qasper.qasper_rawprompt_gen import \
        qasper_datasets
    from opencompass.configs.datasets.ruler.ruler_8k_rawprompt_gen import \
        ruler_datasets as ruler_8k_datasets
    # Summary Groups
    from opencompass.configs.summarizers.groups.babilong import \
        babilong_summary_groups as _babilong_all_groups
    from opencompass.configs.summarizers.groups.longbench import \
        longbench_summary_groups
    from opencompass.configs.summarizers.groups.ruler import \
        ruler_summary_groups as _ruler_all_groups
    from opencompass.configs.summarizers.needlebench import \
        needlebench_8k_summarizer

# The 8K NeedleBench config expands to hundreds of depth/length combinations.
needlebench_8k_datasets = needlebench_8k_datasets[:1]
needlebench_first_abbr = needlebench_8k_datasets[0]['abbr']

# MRCR expands to multiple needle counts and context ranges. The first entry is
# its shortest two-needle subset and is enough to exercise the mock path.
mrcr_datasets = mrcr_datasets[:1]
mrcr_first_abbr = mrcr_datasets[0]['abbr']

datasets = sum(
    (v for k, v in locals().items() if k.endswith('_datasets')),
    [],
)

# The mock endpoint is shared with two other suite processes.
for model in models:
    model['max_workers'] = min(
        int(model.get('max_workers') or MOCK_API_MAX_WORKERS),
        MOCK_API_MAX_WORKERS,
    )

needlebench_8k_summary_groups = [
    dict(
        name='NeedleBench-First-Subset',
        subsets=[needlebench_first_abbr],
    )
]
ruler_8k_summary_groups = [
    group for group in _ruler_all_groups if group['name'] == 'ruler_8k'
]
babilong_0k_summary_groups = [
    group for group in _babilong_all_groups if group['name'] == 'babilong_0k'
]
mrcr_first_summary_groups = [
    dict(
        name='MRCR-v2-First-Subset',
        subsets=[mrcr_first_abbr],
    )
]
summary_groups = sum(
    [v for k, v in locals().items() if k.endswith('_summary_groups')], [])

summarizer = dict(
    dataset_abbrs=[
        ['ruler_8k', 'naive_average'],
        ['NeedleBench-First-Subset', 'naive_average'],
        ['babilong_0k', 'naive_average'],
        ['longbench', 'naive_average'],
        ['LongBenchv2', 'accuracy'],
        ['LongBenchv2', 'accuracy_easy'],
        ['LongBenchv2', 'accuracy_hard'],
        ['LongBenchv2', 'accuracy_short'],
        ['LongBenchv2', 'accuracy_medium'],
        ['LongBenchv2', 'accuracy_long'],
        ['MRCR-v2-First-Subset', 'naive_average'],
        ['QASPER', 'score'],
        ['AA-LCR', 'accuracy'],
    ],
    summary_groups=summary_groups,
)

obj_llm_judge_cfg = models[0]
for item in datasets:
    try:
        evaluator = item['eval_cfg']['evaluator']
        if 'judge_cfg' in evaluator:
            evaluator['judge_cfg'] = obj_llm_judge_cfg
        elif 'llm_evaluator' in evaluator and 'judge_cfg' in evaluator[
                'llm_evaluator']:
            evaluator['llm_evaluator']['judge_cfg'] = obj_llm_judge_cfg
    except (KeyError, TypeError):
        pass

set_mock_test_range(datasets)
del set_mock_test_range

# Repeated evaluation and an outer test_range interact poorly: repetition is
# applied before sampling, while GenericLLMEvaluator reloads ten source rows.
# Mock regression only needs one deterministic pass through the judge path.
for item in aa_lcr_datasets:
    item['n'] = 1
    item['k'] = 1
