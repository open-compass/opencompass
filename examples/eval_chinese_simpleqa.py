from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.chinese_simpleqa.chinese_simpleqa_gen import csimpleqa_datasets

from opencompass.models import HuggingFacewithChatTemplate
from opencompass.models.openai_api import OpenAI
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.summarizers import DefaultSubjectiveSummarizer
from opencompass.tasks.subjective_eval import SubjectiveEvalTask

# -------------Inference Stage ----------------------------------------
models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='Qwen2.5-1.5B-Instruct',
        path='Qwen/Qwen2.5-1.5B-Instruct',
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
        ),
        generation_kwargs=dict(do_sample=True, ),
        max_out_len=200,
        max_seq_len=4096,
        batch_size=8,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]

datasets = sum([v for k, v in locals().items() if ('datasets' in k)], [])
summarizer = dict(type=DefaultSubjectiveSummarizer)

# -------------Evalation Stage ----------------------------------------

## ------------- JudgeLLM Configuration

api_meta_template = dict(round=[
    dict(role='SYSTEM', api_role='SYSTEM'),
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
])
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
    partitioner=dict(type=SubjectiveNaivePartitioner,
                     models=models,
                     judge_models=judge_models),
    runner=dict(type=LocalRunner,
                max_num_workers=16,
                task=dict(type=SubjectiveEvalTask)),
)

work_dir = 'outputs/chinese_simpleqa/'
