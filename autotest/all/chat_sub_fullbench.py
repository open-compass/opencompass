from mmengine.config import read_base

from opencompass.summarizers import DefaultSubjectiveSummarizer

with read_base():
    from autotest.all.config import MOCK_API_MAX_WORKERS  # noqa: F401
    from autotest.all.config import models  # noqa: F401
    from autotest.all.config import concurrent_infer as infer  # noqa: F401
    from autotest.all.config import judge_models, set_mock_test_range
    from opencompass.configs.datasets.chinese_simpleqa.chinese_simpleqa_rawprompt_gen import \
        csimpleqa_datasets
    from opencompass.configs.datasets.SimpleQA.simpleqa_rawprompt_gen import \
        simpleqa_datasets
    # Datasets
    from opencompass.configs.datasets.subjective.alignbench.alignbench_v1_1_judgeby_critiquellm_rawprompt import \
        alignbench_datasets
    from opencompass.configs.datasets.subjective.arena_hard.arena_hard_compare_rawprompt import \
        arenahard_datasets
    from opencompass.configs.datasets.subjective.fofo.fofo_bilingual_judge_rawprompt import \
        fofo_datasets
    from opencompass.configs.datasets.subjective.followbench.followbench_llmeval_rawprompt import \
        followbench_llmeval_datasets
    from opencompass.configs.datasets.subjective.multiround.mtbench101_judge_new_rawprompt import \
        mtbench101_datasets
    from opencompass.configs.datasets.subjective.wildbench.wildbench_pair_judge_rawprompt import \
        wildbench_datasets

datasets = []

for model in models + judge_models:
    model['max_workers'] = min(
        int(model.get('max_workers') or MOCK_API_MAX_WORKERS),
        MOCK_API_MAX_WORKERS,
    )

datasets += sum(
    (v for k, v in locals().items() if k.endswith('_datasets')
     and 'mtbench101' not in k and 'wildbench' not in k),
    [],
)

datasets += mtbench101_datasets
datasets += wildbench_datasets
set_mock_test_range(datasets)
del set_mock_test_range

summary_groups = []

summary_groups.append({
    'name':
    'FoFo',
    'subsets': [['fofo_test_prompts', 'overall'],
                ['fofo_test_prompts_cn', 'overall']],
})
summary_groups.append({
    'name':
    'Followbench',
    'subsets': [
        ['followbench_llmeval_en', 'HSR_AVG'],
        ['followbench_llmeval_en', 'SSR_AVG'],
    ],
})

# Summarizer
summarizer = dict(
    dataset_abbrs=[
        ['alignment_bench_v1_1', '总分'],
        ['arenahard', 'score'],
        ['Followbench', 'naive_average'],
        ['FoFo', 'naive_average'],
        ['mtbench101', 'avg'],
        ['wildbench', 'average'],
        ['simpleqa', 'accuracy_given_attempted'],
        ['chinese_simpleqa', 'given_attempted_accuracy'],
        '',
        ['alignment_bench_v1_1', '专业能力'],
        ['alignment_bench_v1_1', '数学计算'],
        ['alignment_bench_v1_1', '基本任务'],
        ['alignment_bench_v1_1', '逻辑推理'],
        ['alignment_bench_v1_1', '中文理解'],
        ['alignment_bench_v1_1', '文本写作'],
        ['alignment_bench_v1_1', '角色扮演'],
        ['alignment_bench_v1_1', '综合问答'],
        ['fofo_test_prompts', 'overall'],
        ['fofo_test_prompts_cn', 'overall'],
        ['followbench_llmeval_en', 'HSR_AVG'],
        ['followbench_llmeval_en', 'SSR_AVG'],
        ['followbench_llmeval_en', 'HSR_L1'],
        ['followbench_llmeval_en', 'HSR_L2'],
        ['followbench_llmeval_en', 'HSR_L3'],
        ['followbench_llmeval_en', 'HSR_L4'],
        ['followbench_llmeval_en', 'HSR_L5'],
        ['followbench_llmeval_en', 'SSR_L1'],
        ['followbench_llmeval_en', 'SSR_L2'],
        ['followbench_llmeval_en', 'SSR_L3'],
        ['followbench_llmeval_en', 'SSR_L4'],
        ['followbench_llmeval_en', 'SSR_L5'],
        ['simpleqa', 'f1'],
    ],
    type=DefaultSubjectiveSummarizer,
    summary_groups=summary_groups,
)
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks.subjective_eval import SubjectiveEvalTask

eval = dict(
    partitioner=dict(type=SubjectiveNaivePartitioner,
                     models=models,
                     judge_models=judge_models),
    runner=dict(type=LocalRunner,
                max_num_workers=16,
                task=dict(type=SubjectiveEvalTask)),
)
