# flake8: noqa
from mmengine.config import read_base

from opencompass.models import ClaudeSDK, OpenAISDK
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalWatchTask, OpenICLInferConcurrentTask

with read_base():
    from opencompass.configs.datasets.aime2024.aime2024_cascade_eval_rawprompt_gen_2f2c96 import \
        aime2024_datasets
    from opencompass.configs.datasets.aime2025.aime2025_cascade_eval_rawprompt_gen_2f2c96 import \
        aime2025_datasets
    from opencompass.configs.datasets.aime2026.aime2026_cascade_eval_rawprompt_gen_0970dd import \
        aime2026_datasets

datasets = sum(
    (v for k, v in locals().items() if k.endswith('_datasets')),
    [],
)

models = [
    dict(
        abbr='claude-sdk-opus45-reasoning',
        type=ClaudeSDK,
        path='claude-opus-4-5-20251101',
        key='ENV',
        query_per_second=4,
        batch_size=16,
        temperature=1.0,
        max_out_len=63999,
        max_seq_len=100000,
        retry=10,
        stream=False,
        thinking=dict(type='enabled', budget_tokens=60000),
    ),
]

judge_cfg = dict(
    abbr='gpt-4o-judge',
    type=OpenAISDK,
    path='gpt-4o',
    mode='front',
    query_per_second=5,
    batch_size=64,
    temperature=0.001,
    tokenizer_path='gpt-4o',
    max_out_len=4000,
    max_seq_len=24000,
    retry=10,
)

for item in datasets:
    evaluator = item['eval_cfg']['evaluator']
    if 'judge_cfg' in evaluator:
        evaluator['judge_cfg'] = judge_cfg
    elif 'llm_evaluator' in evaluator.keys() and \
            'judge_cfg' in evaluator['llm_evaluator']:
        evaluator['llm_evaluator']['judge_cfg'] = judge_cfg

summarizer = dict()

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=8),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLInferConcurrentTask),
        max_num_workers=64,
        keep_tmp_file=True,
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner, n=8),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLEvalWatchTask),
        max_num_workers=32,
        keep_tmp_file=True,
    ),
)

work_dir = 'outputs/claude_sdk_oc_reasoning'
