from mmengine.config import read_base

with read_base():
    from .datasets.subjective.compassbench.compassbench_checklist import (
        checklist_datasets,
    )
from opencompass.partitioners import NaivePartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask

from opencompass.summarizers.subjective.compassbench_v13 import CompassBenchSummarizer
from opencompass.models import HuggingFacewithChatTemplate
from opencompass.models import TurboMindModelwithChatTemplate

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ]
)
models = [
    # Choose different engines to start the job
    # dict(
    #     type=HuggingFacewithChatTemplate,
    #     abbr="internlm2-chat-1.8b",
    #     path="internlm/internlm2-chat-1_8b-sft",
    #     max_out_len=1024,
    #     batch_size=8,
    #     run_cfg=dict(num_gpus=1),
    # ),
    # dict(
    #     type=HuggingFacewithChatTemplate,
    #     abbr="gpt4o",
    #     path="internlm/internlm2-chat-1_8b-sft",
    #     max_out_len=1024,
    #     batch_size=8,
    #     run_cfg=dict(num_gpus=1),
    # ),
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='internlm2-chat-1.8b-turbomind',
        path='internlm/internlm2-chat-1_8b',
        engine_config=dict(session_len=7168, max_batch_size=16, tp=1),
        gen_config=dict(top_k=1000, temperature=1, top_p=0.9, max_new_tokens=2048),
        max_seq_len=7168,
        max_out_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
    ),
    # Mock as gpt4o
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='gpt4o',
        path='internlm/internlm2-chat-1_8b',
        engine_config=dict(session_len=7168, max_batch_size=16, tp=1),
        gen_config=dict(top_k=1000, temperature=1, top_p=0.9, max_new_tokens=2048),
        max_seq_len=7168,
        max_out_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
    ),
]
# -------------Inference Stage ----------------------------------------
# For subjective evaluation, we often set do sample for models
datasets = [*checklist_datasets]
infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner, max_num_workers=16, task=dict(type=OpenICLInferTask)),
)
# -------------Evalation Stage ----------------------------------------
## ------------- JudgeLLM Configuration
judge_models = [models[0]]
## ------------- Evaluation Configuration
eval = dict(
    partitioner=dict(
        type=SubjectiveNaivePartitioner,
        models=models,
        judge_models=judge_models,
    ),
    runner=dict(
        type=LocalRunner, max_num_workers=16, task=dict(type=SubjectiveEvalTask)
    ),
)
summarizer = dict(type=CompassBenchSummarizer)
work_dir = 'outputs/debug_checklist/'
