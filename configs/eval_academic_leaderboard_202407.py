from mmengine.config import read_base
import os.path as osp
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask


#######################################################################
#                          PART 0  Essential Configs                  #
#######################################################################
with read_base():
    # Datasets Part
    ## Core Set
    # ## Examination
    from opencompass.configs.datasets.mmlu.mmlu_openai_simple_evals_gen_b618ea import mmlu_datasets
    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_0shot_cot_gen_08c1de import mmlu_pro_datasets
    from opencompass.configs.datasets.cmmlu.cmmlu_0shot_cot_gen_305931 import cmmlu_datasets
    # ## Reasoning
    from opencompass.configs.datasets.bbh.bbh_gen_4a31fa import bbh_datasets
    from opencompass.configs.datasets.gpqa.gpqa_openai_simple_evals_gen_5aeece import gpqa_datasets
    # ## Math
    from opencompass.configs.datasets.math.math_0shot_gen_393424 import math_datasets
    # ## Coding
    from opencompass.configs.datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets
    # ## Instruction Following
    from opencompass.configs.datasets.IFEval.IFEval_gen_3321a3 import ifeval_datasets

    # Summarizer
    from opencompass.configs.summarizers.groups.mmlu import mmlu_summary_groups
    from opencompass.configs.summarizers.groups.mmlu_pro import mmlu_pro_summary_groups
    from opencompass.configs.summarizers.groups.cmmlu import cmmlu_summary_groups
    from opencompass.configs.summarizers.groups.bbh import bbh_summary_groups


    # Model List
    # from opencompass.configs.models.qwen.lmdeploy_qwen2_1_5b_instruct import models as lmdeploy_qwen2_1_5b_instruct_model
    # from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_5_7b_chat import models as hf_internlm2_5_7b_chat_model
    # from opencompass.configs.models.openbmb.hf_minicpm_2b_sft_bf16 import models as hf_minicpm_2b_sft_bf16_model
    # from opencompass.configs.models.yi.hf_yi_1_5_6b_chat import models as hf_yi_1_5_6b_chat_model
    # from opencompass.configs.models.gemma.hf_gemma_2b_it import models as hf_gemma_2b_it_model
    # from opencompass.configs.models.yi.hf_yi_1_5_34b_chat import models as hf_yi_1_5_34b_chat_model

#######################################################################
#                          PART 1  Datasets List                      #
#######################################################################
# datasets list for evaluation
datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])


#######################################################################
#                       PART 2  Datset Summarizer                     #
#######################################################################
# with read_base():

core_summary_groups = [
    {
        'name': 'core_average',
        'subsets': [
            ['mmlu', 'accuracy'],
            ['mmlu_pro', 'accuracy'],
            # ['cmmlu', 'naive_average'],
            ['cmmlu', 'accuracy'],
            ['bbh', 'score'],
            ['math', 'accuracy'],
            ['openai_humaneval', 'humaneval_pass@1'],
            ['GPQA_diamond', 'accuracy'],
            ['IFEval', 'Prompt-level-strict-accuracy'],
        ],
    },
]

summarizer = dict(
    dataset_abbrs=[
        ['core_average', 'naive_average'],
        ['mmlu', 'accuracy'],
        ['mmlu_pro', 'accuracy'],
        ['cmmlu', 'accuracy'],
        ['bbh', 'score'],
        ['math', 'accuracy'],
        ['openai_humaneval', 'humaneval_pass@1'],
        ['GPQA_diamond', 'accuracy'],
        ['IFEval', 'Prompt-level-strict-accuracy'],
        '',

        ['mmlu', 'accuracy'],
        ['mmlu-stem', 'accuracy'],
        ['mmlu-social-science', 'accuracy'],
        ['mmlu-humanities', 'accuracy'],
        ['mmlu-other', 'accuracy'],

        '',
        ['mmlu_pro', 'accuracy'],
        ['mmlu_pro_math','accuracy'],
        ['mmlu_pro_physics', 'accuracy'],
        ['mmlu_pro_chemistry', 'accuracy'],
        ['mmlu_pro_law', 'accuracy'],
        ['mmlu_pro_engineering', 'accuracy'],
        ['mmlu_pro_other', 'accuracy'],
        ['mmlu_pro_economics', 'accuracy'],
        ['mmlu_pro_health', 'accuracy'],
        ['mmlu_pro_psychology', 'accuracy'],
        ['mmlu_pro_business', 'accuracy'],
        ['mmlu_pro_biology', 'accuracy'],
        ['mmlu_pro_philosophy', 'accuracy'],
        ['mmlu_pro_computer_science','accuracy'],
        ['mmlu_pro_history', 'accuracy'],
        '',
        ['cmmlu', 'accuracy'],
        ['cmmlu-stem', 'accuracy'],
        ['cmmlu-social-science', 'accuracy'],
        ['cmmlu-humanities', 'accuracy'],
        ['cmmlu-other', 'accuracy'],
        ['cmmlu-china-specific', 'accuracy'],
        '',
        ['bbh', 'extract_rate'],
        ['math', 'extract_rate'],
        # ['openai_humaneval', 'extract_rate'],
        ['GPQA_diamond', 'extract_rate'],
        # ['IFEval', 'extract_rate'],
        '',
        ['mmlu', 'extract_rate'],
        ['mmlu-stem', 'extract_rate'],
        ['mmlu-social-science', 'extract_rate'],
        ['mmlu-humanities', 'extract_rate'],
        ['mmlu-other', 'extract_rate'],
        '',
        ['mmlu_pro', 'extract_rate'],
        ['mmlu_pro_math', 'extract_rate'],
        ['mmlu_pro_physics', 'extract_rate'],
        ['mmlu_pro_chemistry', 'extract_rate'],
        ['mmlu_pro_law', 'extract_rate'],
        ['mmlu_pro_engineering', 'extract_rate'],
        ['mmlu_pro_other', 'extract_rate'],
        ['mmlu_pro_economics', 'extract_rate'],
        ['mmlu_pro_health', 'extract_rate'],
        ['mmlu_pro_psychology', 'extract_rate'],
        ['mmlu_pro_business', 'extract_rate'],
        ['mmlu_pro_biology', 'extract_rate'],
        ['mmlu_pro_philosophy', 'extract_rate'],
        ['mmlu_pro_computer_science', 'extract_rate'],
        ['mmlu_pro_history', 'extract_rate'],
        '',
        ['cmmlu', 'extract_rate'],
        ['cmmlu-stem', 'extract_rate'],
        ['cmmlu-social-science', 'extract_rate'],
        ['cmmlu-humanities', 'extract_rate'],
        ['cmmlu-other', 'extract_rate'],
        ['cmmlu-china-specific', 'extract_rate'],

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
    partitioner=dict(
        type=NumWorkerPartitioner,
        num_worker=8
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=16,
        retry=0, # Modify if needed
        task=dict(type=OpenICLInferTask)
    ),
)

# eval with local runner
eval = dict(
    partitioner=dict(type=NaivePartitioner, n=10),
    runner=dict(
        type=LocalRunner,
        max_num_workers=16,
        task=dict(type=OpenICLEvalTask)),
)


#######################################################################
#                      PART 5  Utils Configuaration                   #
#######################################################################
base_exp_dir = 'outputs/corebench_v1_9/'
work_dir = osp.join(base_exp_dir, 'chat_objective')
