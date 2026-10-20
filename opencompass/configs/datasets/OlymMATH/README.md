# OlymMATH
[GitHub Link](https://github.com/RUCAIBox/OlymMATH)

Dataset OlymMATH, please refer to the paper:
Challenging the Boundaries of Reasoning: An Olympiad-Level Math Benchmark for Large Language Models by Haoxiang Sun, Yingqian Min, Zhipeng Chen, Wayne Xin Zhao, Zheng Liu, Zhongyuan Wang, Lei Fang, and Ji-Rong Wen.


## How to eval OlymMATH with model judge
This is a simple example:
```python

from opencompass.models import OpenAISDK, OpenAI
from mmengine.config import read_base


with read_base():
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_7b_instruct import models as qwen2_5_7b_instruct_model
    from opencompass.configs.datasets.OlymMATH.olymmath_gen import olymmath_datasets

##################  Judge Config  ##################
api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
], )

judge_cfg = dict(
    # An API model with OpenAI API format is required for Judge
        abbr='qwen2-5-32B-Instruct',
        type=OpenAISDK,
        path='Qwen/Qwen2.5-32B-Instruct',
        key='sk-1234',
        openai_api_base=[
            'http://172.30.56.1:4000/v1',
        ],
        meta_template=api_meta_template,
        query_per_second=16,
        batch_size=1024,
        temperature=0.001,
        max_completion_tokens=32768,
        tokenizer_path='gpt-4o-2024-05-13',
        verbose=True,
        max_out_len=16384,
        max_seq_len=32768,
)

##################  Model Config  ##################
models = [*qwen2_5_7b_instruct_model]

##################  Dataset Config  ##################
datasets = [*olymmath_datasets]

# Set judge_cfg for evaluation
for item in datasets:
    item['infer_cfg']['inferencer']['max_out_len'] = 32768
    if 'judge_cfg' in item['eval_cfg']['evaluator']:
        item['eval_cfg']['evaluator']['judge_cfg'] = judge_cfg


work_dir = './outputs/olymmath_llm_eval'
```
