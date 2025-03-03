# Support AIME-2024 with Repeat8
# Support MATH-500
# Support OlympiadBench
# Support OmniMath
# Support LiveMathBench-202412-Hard

import os.path as osp
from itertools import product
from opencompass.models import OpenAISDK
from mmengine.config import read_base
from opencompass.utils.text_postprocessors import extract_non_reasoning_content
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.runners import LocalRunner
from opencompass.models import (
    TurboMindModelwithChatTemplate,
)

#######################################################################
#                          PART 1  Datasets List                      #
#######################################################################
with read_base():
    # You can comment out the datasets you don't want to evaluate

    # Datasets
    # from opencompass.configs.datasets.math.math_prm800k_500_llmverify_gen_6ff468 import math_datasets # 1 Run
    from opencompass.configs.datasets.aime2024.aime2024_llmverify_repeat8_gen_e8fcee import aime2024_datasets # 8 Run
    # from opencompass.configs.datasets.OlympiadBench.OlympiadBench_0shot_llmverify_gen_be8b13 import olympiadbench_datasets
    # from opencompass.configs.datasets.omni_math.omni_math_llmverify_gen_ccf9c0 import omnimath_datasets # 1 Run
    # from opencompass.configs.datasets.livemathbench.livemathbench_hard_custom_llmverify_gen_85d0ef import livemathbench_datasets


    # Summarizer
    from opencompass.configs.summarizers.groups.OlympiadBench import OlympiadBenchMath_summary_groups

datasets = sum(
    (v for k, v in locals().items() if k.endswith('_datasets')),
    [],
)

# Set LLM Verifier used for each dataset

verifier_cfg = dict(
        abbr='qwen2-5-32B-Instruct',
        type=OpenAISDK,
        path='Qwen/Qwen2.5-32B-Instruct', # You need to set your own judge model path
        key='sk-1234', # You need to set your own API key
        openai_api_base=[
            'http://172.30.56.1:4000/v1', # You need to set your own API base
        ],
        meta_template=dict(
            round=[
                dict(role='HUMAN', api_role='HUMAN'),
                dict(role='BOT', api_role='BOT', generate=True),
            ], 
        ),
        query_per_second=16,
        batch_size=1024,
        temperature=0.001,
        tokenizer_path='gpt-4o-2024-05-13',
        verbose=True,
        max_out_len=16384,
        # max_seq_len=32768,
        max_seq_len=49152,
)

for item in datasets:
    # item['infer_cfg']['inferencer']['max_out_len'] = 32768 # You can unset this line if you want to avoid length cutoff
    if 'judge_cfg' in item['eval_cfg']['evaluator']:
        item['eval_cfg']['evaluator']['judge_cfg'] = verifier_cfg


#######################################################################
#                          PART 2  Model List                         #
#######################################################################

models = sum([v for k, v in locals().items() if k.endswith('_model')], [])

models += [
    # You can comment out the models you don't want to evaluate
    # All models use sampling mode
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='deepseek-r1-distill-qwen-7b-turbomind',
        path='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
        engine_config=dict(session_len=32768, max_batch_size=128, tp=1),
        gen_config=dict(
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.95,
                        max_new_tokens=32768),
        max_seq_len=32768,
        max_out_len=32768,
        batch_size=64,
        run_cfg=dict(num_gpus=1),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    ),
    # dict(
    #     type=TurboMindModelwithChatTemplate,
    #     abbr='deepseek-r1-distill-qwen-14b-turbomind',
    #     path='deepseek-ai/DeepSeek-R1-Distill-Qwen-14B',
    #     engine_config=dict(session_len=32768, max_batch_size=128, tp=2),
    #     gen_config=dict(
    #                     do_sample=True,
    #                     temperature=0.6,
    #                     top_p=0.95,
    #                     max_new_tokens=32768),
    #     max_seq_len=32768,
    #     max_out_len=32768,
    #     batch_size=128,
    #     run_cfg=dict(num_gpus=2),
    #     pred_postprocessor=dict(type=extract_non_reasoning_content)
    # ),
    # dict(
    #     type=TurboMindModelwithChatTemplate,
    #     abbr='deepseek-r1-distill-qwen-32b-turbomind',
    #     path='deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
    #     engine_config=dict(session_len=32768, max_batch_size=128, tp=4),
    #     gen_config=dict(
    #                     do_sample=True,
    #                     temperature=0.6,
    #                     top_p=0.95,
    #                     max_new_tokens=16384),
    #     max_seq_len=32768,
    #     max_out_len=16384,
    #     batch_size=128,
    #     run_cfg=dict(num_gpus=4),
    #     pred_postprocessor=dict(type=extract_non_reasoning_content)
    # ),
]

#######################################################################
#                          PART 3  Inference/Evaluation               #
#######################################################################

# Inference configuration
infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        num_worker=1
        # Similar with data-parallelism, how many workers for evaluation,
        # each worker will evaluate a part of the dataset. Total GPUs = num_worker * num_gpus_per_worker
        # For example, If you have 8 GPUs, for 7B model using 1 GPU for one instance, you can set num_worker=8
        # to max-utilize the GPUs.
        # If you have 8 GPUs, for 14B model using 2 GPUs for one instance, you can set num_worker=4
    ),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLInferTask)
    ),
)

# Evaluation configuration
eval = dict(
    partitioner=dict(
        type=NaivePartitioner, n=8
    ),
    runner=dict(
        type=LocalRunner,
        task=dict(
            type=OpenICLEvalTask)
    ),
)


#######################################################################
#                          PART 4  Summarizer                         #
#######################################################################


summary_groups = sum(
    [v for k, v in locals().items() if k.endswith('_summary_groups')], []
)

summary_groups.extend([
    {
        'name': 'AIME2024-Aveage8',
        'subsets':[[f'aime2024-run{idx}', 'accuracy'] for idx in range(8)]
    },
    {
        'name': 'LiveMathBench-v202412-Hard-Aveage8',
        'subsets':[[
            f'livemathbench_hard_custom_{split}_run{run_idx}', 'accuracy'] 
                for split, run_idx in product(['hard_cn', 'hard_en'], range(8))
        ]
    }
])

# Summarizer
summarizer = dict(
    dataset_abbrs=[
        'MATH',
        # ['LiveMathBench-k1-n1', 'pass@1'],
        # ['LiveMathBench-v202412-greedy', 'G-Pass@1_0.0'],
        # ['aime2024', 'accuracy'],
        ['math_prm800k_500-llmjudge', 'accuracy'],
        ['AIME2024-Aveage8', 'naive_average'],
        ['LiveMathBench-v202412-Hard-Aveage8', 'naive_average'],
        ['OlympiadBenchMath', 'accuracy'],
        ['OmniMath', 'accuracy'],
    ],
    summary_groups=summary_groups,
)


#######################################################################
#                          PART 5  Utils                              #
#######################################################################

work_dir = 'outputs/deepseek_r1_reasoning'


