from mmengine.config import read_base

with read_base():
    from .datasets.subjective.alignbench.alignbench_judgeby_critiquellm import subjective_datasets

from opencompass.models import HuggingFaceCausalLM, HuggingFace, OpenAIAllesAPIN, HuggingFaceChatGLM3
from opencompass.partitioners import NaivePartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.runners import SlurmSequentialRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask
from opencompass.summarizers import AlignmentBenchSummarizer

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ]
)

# -------------Inference Stage ----------------------------------------
# For subjective evaluation, we often set do sample for models
models = [
    dict(
        type=HuggingFaceChatGLM3,
        abbr='chatglm3-6b-hf',
        path='THUDM/chatglm3-6b',
        tokenizer_path='THUDM/chatglm3-6b',
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
        ),
        generation_kwargs=dict(
            do_sample=True,
        ),
        meta_template=api_meta_template,
        max_out_len=2048,
        max_seq_len=4096,
        batch_size=1,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]

datasets = [*subjective_datasets]

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=SlurmSequentialRunner,
        partition='llmeval',
        quotatype='auto',
        max_num_workers=256,
        task=dict(type=OpenICLInferTask),
    ),
)

# -------------Evalation Stage ----------------------------------------

## ------------- JudgeLLM Configuration
judge_models = [dict(
    type=HuggingFaceCausalLM,
    abbr='pandalm-7b-v1-hf',
    path='WeOpenML/PandaLM-7B-v1',
    tokenizer_path='WeOpenML/PandaLM-7B-v1',
    tokenizer_kwargs=dict(
        padding_side='left',
        truncation_side='left',
        trust_remote_code=True,
        use_fast=False,
    ),
    max_out_len=512,
    max_seq_len=2048,
    batch_size=8,
    model_kwargs=dict(device_map='auto', trust_remote_code=True),
    run_cfg=dict(num_gpus=1, num_procs=1),
)]

## ------------- Evaluation Configuration
eval = dict(
    partitioner=dict(type=SubjectiveNaivePartitioner, mode='singlescore', models=models, judge_models=judge_models),
    runner=dict(type=LocalRunner, max_num_workers=2, task=dict(type=SubjectiveEvalTask)),
)

summarizer = dict(type=AlignmentBenchSummarizer)

work_dir = 'outputs/pandalm'
