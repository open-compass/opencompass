from mmengine.config import read_base

from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks.subjective_eval import SubjectiveEvalTask

with read_base():
    from autotest.all.config import \
        concurrent_infer as infer  # noqa: F401, E501
    from autotest.all.config import judge_models, models  # noqa: F401, E501
    from opencompass.configs.datasets.chinese_simpleqa.chinese_simpleqa_gen import \
        csimpleqa_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.SimpleQA.simpleqa_gen_0283c3 import \
        simpleqa_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.subjective.alignbench.alignbench_v1_1_judgeby_critiquellm_new import \
        alignbench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.subjective.alpaca_eval.alpacav2_judgeby_gpt4_new import \
        alpacav2_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.subjective.arena_hard.arena_hard_compare_new import \
        arenahard_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.subjective.fofo.fofo_bilingual_judge_new import \
        fofo_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.subjective.followbench.followbench_llmeval_new import \
        followbench_llmeval_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.subjective.multiround.mtbench101_judge_new import \
        mtbench101_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.subjective.wildbench.wildbench_pair_judge_new import \
        wildbench_datasets  # noqa: F401, E501

datasets = sum(
    (v for k, v in locals().items() if k.endswith('_datasets')
     and 'mtbench101' not in k and 'wildbench' not in k),
    [],
)

datasets += mtbench101_datasets
datasets += wildbench_datasets

eval = dict(
    partitioner=dict(type=SubjectiveNaivePartitioner,
                     models=models,
                     judge_models=judge_models),
    runner=dict(type=LocalRunner,
                max_num_workers=16,
                task=dict(type=SubjectiveEvalTask)),
)
