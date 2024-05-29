from os import getenv as gv
from opencompass.models import HuggingFaceCausalLM
from mmengine.config import read_base

with read_base():
    from .datasets.subjective.compassbench.compassbench_compare import subjective_datasets

from opencompass.models import HuggingFaceCausalLM, HuggingFace, HuggingFaceChatGLM3, OpenAI
from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.partitioners.sub_size import SubjectiveSizePartitioner
from opencompass.runners import LocalRunner
from opencompass.runners import SlurmSequentialRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask
from opencompass.summarizers import CompassBenchSummarizer

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
    reserved_roles=[dict(role='SYSTEM', api_role='SYSTEM')],
)

# -------------Inference Stage ----------------------------------------

from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='internlm2-chat-7b-hf',
        path='internlm/internlm2-chat-7b',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
        stop_words=['</s>', '<|im_end|>'],
        generation_kwargs=dict(
            do_sample=True,
        ),
    )
]

datasets = [*subjective_datasets]

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=SlurmSequentialRunner,
        partition='llmeval',
        quotatype='reserved',
        max_num_workers=256,
        task=dict(type=OpenICLInferTask),
    ),
)

gpt4 = dict(
    abbr='gpt4-turbo',
    type=OpenAI,
    path='gpt-4-1106-preview',
    key='',  # The key will be obtained from $OPENAI_API_KEY, but you can write down your key here as well
    meta_template=api_meta_template,
    query_per_second=1,
    max_out_len=2048,
    max_seq_len=4096,
    batch_size=4,
    retry=20,
    temperature=1,
)  # Re-inference gpt4's predictions or you can choose to use the pre-commited gpt4's predictions

# -------------Evalation Stage ----------------------------------------

## ------------- JudgeLLM Configuration
judge_models = [dict(
    abbr='GPT4-Turbo',
    type=OpenAI,
    path='gpt-4-1106-preview',
    key='',  # The key will be obtained from $OPENAI_API_KEY, but you can write down your key here as well
    meta_template=api_meta_template,
    query_per_second=1,
    max_out_len=1024,
    max_seq_len=4096,
    batch_size=2,
    retry=20,
    temperature=0,
)]

judge_models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='internlm102b',
        path='/mnt/petrelfs/caomaosong/backup_hwfile/100bjudge_6w_epoch1/hf',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=4),
        stop_words=['</s>', '<|im_end|>'],
    ),
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='internlm102b2',
        path='/mnt/petrelfs/caomaosong/backup_hwfile/100bjudge_6w_epoch1/hf',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=4),
        stop_words=['</s>', '<|im_end|>'],
    ),
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='internlm102b3',
        path='/mnt/petrelfs/caomaosong/backup_hwfile/100bjudge_6w_epoch1/hf',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=4),
        stop_words=['</s>', '<|im_end|>'],
    )
]

## ------------- Evaluation Configuration
eval = dict(
    partitioner=dict(
        type=SubjectiveSizePartitioner,
        strategy='split',
        max_task_size=10000000,
        mode='m2n',
        infer_order='double',
        base_models=[gpt4],
        compare_models=models,
        judge_models=judge_models,
    ),
    runner=dict(type=LocalRunner, max_num_workers=32, task=dict(type=SubjectiveEvalTask)),
    #given_pred = [{'abbr':'gpt4-turbo', 'path':''}]
)

work_dir = 'outputs/compassbench/'

summarizer = dict(type=CompassBenchSummarizer, summary_type='half_add')
