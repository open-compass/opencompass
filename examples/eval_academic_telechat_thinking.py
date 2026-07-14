# flake8: noqa

from mmengine.config import read_base

from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask


#######################################################################
#                          PART 0  Essential Configs                  #
#######################################################################
with read_base():
    # Datasets
    from opencompass.configs.datasets.aime2025.aime2025_llmjudge_academic import \
        aime2025_datasets
    from opencompass.configs.datasets.gpqa.gpqa_cascade_eval_academic import \
        gpqa_datasets
    from opencompass.configs.datasets.IFEval.IFEval_gen_353ae7 import \
        ifeval_datasets
    from opencompass.configs.datasets.livecodebench.livecodebench_v6_academic import \
        LCBCodeGeneration_dataset
    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_0shot_cot_gen_08c1de import \
        mmlu_pro_datasets
    from opencompass.configs.datasets.HLE.hle_llmverify_academic import \
        hle_datasets


    # Summary Groups
    from opencompass.configs.summarizers.groups.mmlu_pro import \
        mmlu_pro_summary_groups

    # Models (add your models here)
    from opencompass.configs.models.telechat.telechat_thinking_streaming_v1 import  models as telechat_thinking_streaming_v1_model

#######################################################################
#                          PART 1  Datasets List                      #
#######################################################################
# datasets list for evaluation
# Only take LCB generation for evaluation

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')),
               []) + [LCBCodeGeneration_dataset]

# LLM judge config: using LLM to evaluate predictions
judge_cfg = dict()

for item in datasets:
    if 'judge_cfg' in item['eval_cfg']['evaluator']:
        item['eval_cfg']['evaluator']['judge_cfg'] = judge_cfg
    if 'llm_evaluator' in item['eval_cfg']['evaluator'].keys() and 'judge_cfg' in item['eval_cfg']['evaluator']['llm_evaluator']:
        item['eval_cfg']['evaluator']['llm_evaluator']['judge_cfg'] = judge_cfg


#######################################################################
#                       PART 2  Datset Summarizer                     #
#######################################################################

core_summary_groups = [
    {
        'name':
        'core_average',
        'subsets': [
            ['IFEval', 'Prompt-level-strict-accuracy'],
            ['hle_llmjudge', 'accuracy'],
            ['aime2025_repeat_32', 'accuracy (32 runs average)'],
            ['GPQA_diamond_repeat_4', 'accuracy (4 runs average)'],
            ['mmlu_pro', 'naive_average'],
            ['lcb_code_generation_repeat_6', 'pass@1 (6 runs average)'],
        ],
    },
]

summarizer = dict(
    dataset_abbrs=[
        ['core_average', 'naive_average'],
        '',
        'Instruction Following',
        ['IFEval', 'Prompt-level-strict-accuracy'],
        '',
        'General Reasoning',
        ['hle_llmjudge', 'accuracy'],
        ['GPQA_diamond_repeat_4', 'accuracy (4 runs average)'],
        '',
        'Math Calculation',
        ['aime2025_repeat_32', 'accuracy (32 runs average)'],
        '',
        'Knowledge',
        ['mmlu_pro', 'naive_average'],
        '',
        'Code',
        ['lcb_code_generation_repeat_6', 'pass@1 (6 runs average)'],
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)

#######################################################################
#                        PART 3  Models  List                         #
#######################################################################

models = sum([v for k, v in locals().items() if k.endswith('_model')], [])

#######################################################################
#                 PART 4  Inference/Evaluation Configuaration         #
#######################################################################

# Inference configuration using a local runner
# - Partitioner: NumWorkerPartitioner splits tasks across 16 workers
# - Runner: LocalRunner executes tasks locally, with a maximum of 1 concurrent worker
# - Task type: OpenICLInferTask
# - Each worker thread processes a batch of 8 samples

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=16),
    runner=dict(
        type=LocalRunner,
        max_num_workers=4,
        task=dict(type=OpenICLInferTask),
    ),
)

# Evaluation configuration using a local runner
# - Partitioner: NaivePartitioner splits tasks into 10 partitions
# - Runner: LocalRunner executes tasks locally, with a maximum of 1 concurrent worker
# - Task type: OpenICLEvalTask
# - Each worker thread processes a batch of 8 samples
eval = dict(
    partitioner=dict(type=NaivePartitioner, n=16),
    runner=dict(type=LocalRunner,
                max_num_workers=4,
                task=dict(type=OpenICLEvalTask)),
)

#######################################################################
#                      PART 5  Utils Configuaration                   #
#######################################################################

work_dir = './outputs/eval_TeleChat_thinking'