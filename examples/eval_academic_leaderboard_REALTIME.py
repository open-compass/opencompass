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
    from opencompass.configs.datasets.aime2025.aime2025_llmjudge_gen_5e9f4f import \
        aime2025_datasets
    from opencompass.configs.datasets.gpqa.gpqa_cascade_eval_gen_772ea0 import \
        gpqa_datasets
    from opencompass.configs.datasets.humaneval.humaneval_openai_sample_evals_gen_dcae0e import \
        humaneval_datasets
    from opencompass.configs.datasets.IFEval.IFEval_gen_353ae7 import \
        ifeval_datasets
    from opencompass.configs.datasets.livecodebench.livecodebench_code_generation_repeat_gen_b5b6c5 import \
        LCBCodeGeneration_dataset
    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_0shot_cot_gen_08c1de import \
        mmlu_pro_datasets
    from opencompass.configs.datasets.HLE.hle_llmverify_gen_6ff468 import \
        hle_datasets

    # Summary Groups
    from opencompass.configs.summarizers.groups.mmlu_pro import \
        mmlu_pro_summary_groups

    # Models (add your models here)
    # from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_5_7b_chat import \
    #     models as hf_internlm2_5_7b_chat_model

#######################################################################
#                          PART 1  Datasets List                      #
#######################################################################
# datasets list for evaluation
# Only take LCB generation for evaluation

aime2025_datasets[0]['n'] = 32
gpqa_datasets[0]['n'] = 4
LCBCodeGeneration_dataset['n'], LCBCodeGeneration_dataset['k'], LCBCodeGeneration_dataset['release_version'] = 6, 1, 'v6'
LCBCodeGeneration_dataset['eval_cfg']['evaluator']['release_version'], LCBCodeGeneration_dataset['eval_cfg']['evaluator']['extractor_version'] = 'v6', 'v2'

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
            ['aime2025', 'accuracy (32 runs average)'],
            ['GPQA_diamond_repeat_4', 'accuracy (4 runs average)'],
            ['mmlu_pro', 'naive_average'],
            ['openai_humaneval', 'humaneval_pass@1'],
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
        ['aime2025', 'accuracy (32 runs average)'],
        '',
        'Knowledge',
        ['mmlu_pro', 'naive_average'],
        '',
        'Code',
        ['openai_humaneval', 'humaneval_pass@1'],
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

# Local Runner
infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=8),
    runner=dict(
        type=LocalRunner,
        max_num_workers=16,
        retry=0,  # Modify if needed
        task=dict(type=OpenICLInferTask),
    ),
)

# eval with local runner
eval = dict(
    partitioner=dict(type=NaivePartitioner, n=10),
    runner=dict(type=LocalRunner,
                max_num_workers=16,
                task=dict(type=OpenICLEvalTask)),
)

#######################################################################
#                      PART 5  Utils Configuaration                   #
#######################################################################

work_dir = './outputs/oc_academic_202507'