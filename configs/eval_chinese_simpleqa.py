from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.chinese_simpleqa.chinese_simpleqa import csimpleqa_datasets, CsimpleqaSummarizer
    from opencompass.configs.models.qwen2_5.qwen2_5.qwen2_5_3b_instruct import models as qwen2_5_3b_instruct_model
from opencompass.models.openai_api import OpenAI
from opencompass.runners import LocalRunner
from opencompass.tasks.subjective_eval import SubjectiveEvalTask
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.models import HuggingFacewithChatTemplate
from opencompass.partitioners import NaivePartitioner

# -------------Inference Stage ----------------------------------------
# For subjective evaluation, we often set do sample for models

models = sum([v for k, v in locals().items() if k.endswith('_model')], [])
datasets = sum([v for k, v in locals().items() if ('datasets' in k)], [])

# -------------Evalation Stage ----------------------------------------

## ------------- JudgeLLM Configuration

api_meta_template = dict(
    round=[
        dict(role='SYSTEM', api_role='SYSTEM'),
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ]
)
judge_models = [
    dict(
        # GPT4o
        abbr='gpt-4o-0513-global',
        type=OpenAI,
        # gpt-4o
        path='gpt-4o-0513-global',
        key='xxx',  # provide OPENAI_API_KEY 
        meta_template=api_meta_template,
        query_per_second=16,
        max_out_len=1000,
        batch_size=8,
        retry=3)
]

## ------------- Evaluation Configuration
eval = dict(
    partitioner=dict(type=SubjectiveNaivePartitioner, models=models, judge_models=judge_models),
    runner=dict(type=LocalRunner, max_num_workers=16, task=dict(type=SubjectiveEvalTask)),
)


summarizer = dict(type=CsimpleqaSummarizer, judge_type='general')
work_dir = 'outputs/chinese_simpleqa/'
