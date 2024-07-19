from mmengine.config import read_base

with read_base():
    from .datasets.subjective.compassbench.compassbench_checklist import (
        checklist_datasets,
    )
from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask
from opencompass.summarizers import SubjectiveSummarizer
from opencompass.models import HuggingFacewithChatTemplate

api_meta_template = dict(
    round=[
        dict(role="HUMAN", api_role="HUMAN"),
        dict(role="BOT", api_role="BOT", generate=True),
    ]
)
models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr="internlm2-chat-1.8b",
        path="internlm/internlm2-chat-1_8b-sft",
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    ),
    dict(
        type=HuggingFacewithChatTemplate,
        abbr="gpt4o",
        path="internlm/internlm2-chat-1_8b-sft",
        max_out_len=1024,
        batch_size=8,
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
# summarizer = dict(type=SubjectiveSummarizer, function='subjective')
work_dir = "outputs/debug_checklist/"
