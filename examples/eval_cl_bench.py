from copy import deepcopy

from mmengine.config import read_base

from opencompass.models import OpenAISDK

with read_base():
    from opencompass.configs.datasets.CLBench.clbench_llmjudge_rawprompt_gen import \
        clbench_datasets

judge_cfg = dict(
    abbr='gpt-5.1-low',
    type=OpenAISDK,
    path='gpt-5.1',
    key='ENV',
    openai_api_base='https://api.openai.com/v1',
    mode='front',
    query_per_second=5,
    batch_size=64,
    tokenizer_path='gpt-4',
    max_out_len=8192,
    max_seq_len=45000,
    retry=50,
    openai_extra_kwargs=dict(reasoning_effort='low'),
)

models = [
    dict(
        abbr='gpt-5.2-high',
        type=OpenAISDK,
        path='gpt-5.2',
        key='ENV',
        openai_api_base='https://api.openai.com/v1',
        mode='front',
        query_per_second=1,
        batch_size=1,
        tokenizer_path='gpt-4',
        max_out_len=8192,
        max_seq_len=45000,
        retry=20,
        openai_extra_kwargs=dict(reasoning_effort='high'),
    ),
]

datasets = deepcopy(clbench_datasets)
datasets[0]['eval_cfg']['evaluator']['judge_cfg'] = judge_cfg

work_dir = 'outputs/clbench_gpt52high_gpt51low_judge'
