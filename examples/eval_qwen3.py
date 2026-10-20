
import os.path as osp
from opencompass.models import OpenAISDK
from mmengine.config import read_base
from opencompass.utils.text_postprocessors import extract_non_reasoning_content
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    from opencompass.configs.datasets.aime2024.aime2024_cascade_eval_gen_5e9f4f import aime2024_datasets
    from opencompass.configs.datasets.aime2025.aime2025_cascade_eval_gen_5e9f4f import aime2025_datasets
    from opencompass.configs.datasets.math.math_500_cascade_eval_gen_6ff468 import math_datasets

#######################################################################
#                          PART 0  Meta Info                          #
#######################################################################


api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
], 
)


judge_cfg = dict(
        abbr='qwen2-5-32B-Instruct',
        type=OpenAISDK,
        path='Qwen/Qwen2.5-32B-Instruct',
        key='sk-1234',
        openai_api_base=[
            'http://x.x.x.x:4000/v1',
        ],
        meta_template=api_meta_template,
        query_per_second=8,
        batch_size=256,
        temperature=0.001,
        # max_completion_tokens=32768,
        tokenizer_path='gpt-4o-2024-05-13',
        # verbose=True,
        max_out_len=16384,
        max_seq_len=32768,
        # max_seq_len=49152,
        mode='mid',
        retry=10
)

#######################################################################
#                          PART 1  Datasets List                      #
#######################################################################

repeated_info = [
    (math_datasets, 4),
    (aime2024_datasets, 32),
    (aime2025_datasets, 32),
]

for datasets_, num in repeated_info:
    for dataset_ in datasets_:
        dataset_['n'] = num

datasets = sum(
    (v for k, v in locals().items() if k.endswith('_datasets')),
    [],
)

for item in datasets:
    item['infer_cfg']['inferencer']['max_out_len'] = 32768
    try:
        if 'judge_cfg' in item['eval_cfg']['evaluator']:
           item['eval_cfg']['evaluator']['judge_cfg'] = judge_cfg
        elif'judge_cfg' in item['eval_cfg']['evaluator']['llm_evaluator']:
            item['eval_cfg']['evaluator']['llm_evaluator']['judge_cfg'] = judge_cfg
    except:
        pass
#######################################################################
#                       PART 2  Dataset Summarizer                    #
#######################################################################

summarizer = dict(
    dataset_abbrs=[
        'MATH',
        ['math_prm800k_500', 'accuracy (4 runs average)'],
        ['aime2024', 'accuracy (32 runs average)'],
        ['aime2025', 'accuracy (32 runs average)'],
        ['livemathbench_hard', 'naive_average'],
        ['OlympiadBenchMath', 'accuracy'],
        ['olymmath', 'naive_average'],
    ],
    summary_groups = sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []
    ),
)

#######################################################################
#                        PART 3  Models  List                         #
#######################################################################
models = sum([v for k, v in locals().items() if k.endswith('_model')], [])
models += [

    dict(
        abbr='Qwen_Qwen3-235B-A22B',
        type=OpenAISDK,
        path='Qwen/Qwen3-235B-A22B',
        key='sk-admin',
        openai_api_base=[
            'http://106.15.231.215:40007/v1/',
        ],
        meta_template=dict(
            # begin=dict(role='SYSTEM', api_role='SYSTEM', prompt=''),
            round=[
                dict(role='HUMAN', api_role='HUMAN'),
                # XXX: all system roles are mapped to human in purpose
                dict(role='BOT', api_role='BOT', generate=True),
            ]
        ),
        query_per_second=16,
        batch_size=128,
        # batch_size=1,
        temperature=0.6,
        # max_completion_tokens=32768,
        tokenizer_path='gpt-4',
        # verbose=True,
        max_out_len=32768,
        max_seq_len=32768,
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    ),
]

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=8),
    runner=dict(type=LocalRunner, task=dict(type=OpenICLInferTask)),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner, n=8),
    runner=dict(type=LocalRunner, task=dict(type=OpenICLEvalTask)),
)

base_exp_dir = 'outputs/qwen3_reasoning'
work_dir = osp.join(base_exp_dir, 'chat_objective')