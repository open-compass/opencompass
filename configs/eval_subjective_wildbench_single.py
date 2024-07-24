from mmengine.config import read_base

with read_base():
    from .datasets.subjective.wildbench.wildbench_single_judge import wildbench_single_datasets

from opencompass.models import HuggingFaceCausalLM, HuggingFace, HuggingFaceChatGLM3, OpenAI
from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.partitioners.sub_size import SubjectiveSizePartitioner
from opencompass.runners import LocalRunner
from opencompass.runners import SlurmSequentialRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask
from opencompass.summarizers import WildBenchSingleSummarizer
from opencompass.models import HuggingFacewithChatTemplate


api_meta_template = dict(
    round=[
        dict(role='SYSTEM', api_role='SYSTEM'),
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ]
)

# -------------Inference Stage ----------------------------------------
# For subjective evaluation, we often set do sample for models
# set max_out_len to 4096.
models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='llama-3-8b-instruct-hf',
        path='meta-llama/Meta-Llama-3-8B-Instruct',
        max_out_len=8192,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
    )
]

for model in models:
    model['generation_kwargs'] = dict(do_sample=True)

datasets = [*wildbench_single_datasets]

# -------------Evalation Stage ----------------------------------------

## ------------- JudgeLLM Configuration
judge_models = [dict(
    abbr='GPT4-Turbo',
    type=OpenAI,
    path='gpt-4-0613', # To compare with the official leaderboard, please use gpt4-0613
    key='xxxx',  # The key will be obtained from $OPENAI_API_KEY, but you can write down your key here as well
    meta_template=api_meta_template,
    query_per_second=16,
    max_out_len=2048,
    max_seq_len=2048,
    batch_size=8,
    temperature=0,
)]


infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=1000, strategy='split'),
    runner=dict(
        type=SlurmSequentialRunner,
        max_num_workers=64,
        quotatype='reserved',
        partition='llmeval',
        task=dict(type=OpenICLInferTask)),
)

## single evaluation
eval = dict(
    partitioner=dict(type=SubjectiveSizePartitioner, strategy='split', max_task_size=10000, mode='singlescore', models=models, judge_models=judge_models),
    runner=dict(type=LocalRunner,
                max_num_workers=2,
                task=dict(type=SubjectiveEvalTask)),
)

summarizer = dict(type=WildBenchSingleSummarizer)

work_dir = 'outputs/wildbench/'
