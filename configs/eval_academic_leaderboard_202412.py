from mmengine.config import read_base
import os.path as osp
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner, VOLCRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask


#######################################################################
#                          PART 0  Essential Configs                  #
#######################################################################
with read_base():
    # Datasets Part
    # Knowledge
    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_0shot_cot_gen_08c1de import (
        mmlu_pro_datasets,
    )
    # General Reasoning
    from opencompass.configs.datasets.gpqa.gpqa_openai_simple_evals_gen_5aeece import (
        gpqa_datasets,
    )
    from opencompass.configs.datasets.bbh.bbh_0shot_nocot_gen_925fc4 import (
        bbh_datasets,
    )
    from opencompass.configs.datasets.humaneval.humaneval_openai_sample_evals_gen_dcae0e import (
        humaneval_datasets,
    )

    # Instruction Following
    from opencompass.configs.datasets.IFEval.IFEval_gen_353ae7 import (
        ifeval_datasets,
    )
    from opencompass.configs.datasets.livecodebench.livecodebench_gen_a4f90b import (
        LCBCodeGeneration_dataset,
    )

    # Math
    from opencompass.configs.datasets.aime2024.aime2024_gen_6e39a4 import (
        aime2024_datasets,
    )
    from opencompass.configs.datasets.math.math_prm800k_500_0shot_cot_gen import (
        math_datasets,
    )

    # Summary Groups
    from opencompass.configs.summarizers.groups.bbh import bbh_summary_groups
    from opencompass.configs.summarizers.groups.mmlu_pro import (
        mmlu_pro_summary_groups,
    )

    # Model List
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_5_7b_chat import (
        models as hf_internlm2_5_7b_chat_model,
    )

#######################################################################
#                          PART 1  Datasets List                      #
#######################################################################
# datasets list for evaluation
# Only take LCB generation for evaluation
datasets = sum(
    (v for k, v in locals().items() if k.endswith('_datasets')), []
) + [LCBCodeGeneration_dataset]

#######################################################################
#                       PART 2  Datset Summarizer                     #
#######################################################################

core_summary_groups = [
    {
        'name': 'core_average',
        'subsets': [
            ['IFEval', 'Prompt-level-strict-accuracy'],
            ['bbh', 'naive_average'],
            ['math_prm800k_500', 'accuracy'],
            ['aime2024', 'accuracy'],
            ['GPQA_diamond', 'accuracy'],
            ['mmlu_pro', 'naive_average'],
            ['openai_humaneval', 'humaneval_pass@1'],
            ['lcb_code_generation', 'pass@1'],
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
        ['bbh', 'naive_average'],
        ['GPQA_diamond', 'accuracy'],
        '',
        'Math Calculation',
        ['math_prm800k_500', 'accuracy'],
        ['aime2024', 'accuracy'],
        '',
        'Knowledge',
        ['mmlu_pro', 'naive_average'],
        '',
        'Code',
        ['openai_humaneval', 'humaneval_pass@1'],
        ['lcb_code_generation', 'pass@1'],
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []
    ),
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
    runner=dict(
        type=LocalRunner, max_num_workers=16, task=dict(type=OpenICLEvalTask)
    ),
)


#######################################################################
#                      PART 5  Utils Configuaration                   #
#######################################################################
work_dir = './outputs/oc_academic_202412'
