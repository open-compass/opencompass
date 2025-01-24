import os.path as osp

from mmengine.config import read_base

from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

#######################################################################
#                          PART 0  Essential Configs                  #
#######################################################################
with read_base():
    # Datasets Part
    ## Core Set
    # ## Examination
    # ## Reasoning
    from opencompass.configs.datasets.bbh.bbh_gen_98fba6 import bbh_datasets
    from opencompass.configs.datasets.cmmlu.cmmlu_ppl_041cbf import \
        cmmlu_datasets
    from opencompass.configs.datasets.drop.drop_gen_a2697c import drop_datasets
    # ## Scientific
    from opencompass.configs.datasets.gpqa.gpqa_few_shot_ppl_2c9cd6 import \
        gpqa_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_17d0dc import \
        gsm8k_datasets
    from opencompass.configs.datasets.hellaswag.hellaswag_10shot_ppl_59c85e import \
        hellaswag_datasets
    # ## Coding
    from opencompass.configs.datasets.humaneval.deprecated_humaneval_gen_d2537e import \
        humaneval_datasets
    # ## Math
    from opencompass.configs.datasets.math.math_4shot_base_gen_43d5b6 import \
        math_datasets
    from opencompass.configs.datasets.MathBench.mathbench_2024_few_shot_mixed_4a3fd4 import \
        mathbench_datasets
    from opencompass.configs.datasets.mbpp.sanitized_mbpp_gen_742f0c import \
        sanitized_mbpp_datasets
    from opencompass.configs.datasets.mmlu.mmlu_ppl_ac766d import mmlu_datasets
    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_few_shot_gen_bfaf90 import \
        mmlu_pro_datasets
    # Model List
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_1_5b import \
        models as lmdeploy_qwen2_5_1_5b_model
    from opencompass.configs.summarizers.groups.bbh import bbh_summary_groups
    from opencompass.configs.summarizers.groups.cmmlu import \
        cmmlu_summary_groups
    from opencompass.configs.summarizers.groups.mathbench_v1_2024 import \
        mathbench_2024_summary_groups
    # TODO: Add LiveCodeBench
    # ## Instruction Following
    # from opencompass.configs.datasets.IFEval.IFEval_gen_3321a3 import ifeval_datasets
    # Summarizer
    from opencompass.configs.summarizers.groups.mmlu import mmlu_summary_groups
    from opencompass.configs.summarizers.groups.mmlu_pro import \
        mmlu_pro_summary_groups

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
        'name':
        'core_average',
        'subsets': [['mmlu', 'accuracy'], ['mmlu_pro', 'accuracy'],
                    ['cmmlu', 'accuracy'], ['bbh', 'naive_average'],
                    ['hellaswag', 'accuracy'], ['drop', 'accuracy'],
                    ['math', 'accuracy'], ['gsm8k', 'accuracy'],
                    ['mathbench-t (average)', 'naive_average'],
                    ['GPQA_diamond', 'accuracy'],
                    ['openai_humaneval', 'humaneval_pass@1'],
                    ['IFEval', 'Prompt-level-strict-accuracy'],
                    ['sanitized_mbpp', 'score'],
                    ['mathbench-t (average)', 'naive_average']],
    },
]

summarizer = dict(
    dataset_abbrs=[
        ['mmlu', 'accuracy'],
        ['mmlu_pro', 'accuracy'],
        ['cmmlu', 'accuracy'],
        ['bbh', 'naive_average'],
        ['hellaswag', 'accuracy'],
        ['drop', 'accuracy'],
        ['math', 'accuracy'],
        ['gsm8k', 'accuracy'],
        ['mathbench-t (average)', 'naive_average'],
        ['GPQA_diamond', 'accuracy'],
        ['openai_humaneval', 'humaneval_pass@1'],
        ['IFEval', 'Prompt-level-strict-accuracy'],
        ['sanitized_mbpp', 'score'],
        'mathbench-a (average)',
        'mathbench-t (average)'
        '',
        ['mmlu', 'accuracy'],
        ['mmlu-stem', 'accuracy'],
        ['mmlu-social-science', 'accuracy'],
        ['mmlu-humanities', 'accuracy'],
        ['mmlu-other', 'accuracy'],
        '',
        ['mmlu_pro', 'accuracy'],
        ['mmlu_pro_math', 'accuracy'],
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
        ['mmlu_pro_computer_science', 'accuracy'],
        ['mmlu_pro_history', 'accuracy'],
        '',
        ['cmmlu', 'accuracy'],
        ['cmmlu-stem', 'accuracy'],
        ['cmmlu-social-science', 'accuracy'],
        ['cmmlu-humanities', 'accuracy'],
        ['cmmlu-other', 'accuracy'],
        ['cmmlu-china-specific', 'accuracy'],
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
        task=dict(type=OpenICLInferTask)),
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
base_exp_dir = 'outputs/corebench_2409_objective/'
work_dir = osp.join(base_exp_dir, 'base_objective')
