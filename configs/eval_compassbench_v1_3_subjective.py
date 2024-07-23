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
from opencompass.models import OpenAI


api_meta_template = dict(
    round=[
        dict(role="HUMAN", api_role="HUMAN"),
        dict(role="BOT", api_role="BOT", generate=True),
    ],
    reserved_roles=[dict(role="SYSTEM", api_role="SYSTEM")],
)

models = [
    # True GPT4-O
    dict(
        abbr="gpt4o",
        type=OpenAI,
        path="gpt-4o",
        rpm_verbose=True,
        meta_template=api_meta_template,
        key="sk-proj-vacclFdAuEHpe76Cc06iT3BlbkFJqcoZyBVpZOt1MNnJs3Zi",
        query_per_second=1,
        max_out_len=2048,
        max_seq_len=2048,
        batch_size=8,
        retry=10,
    ),
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr="internlm2-chat-1.8b-turbomind",
        path="internlm/internlm2-chat-1_8b",
        engine_config=dict(session_len=7168, max_batch_size=16, tp=1),
        gen_config=dict(top_k=1000, temperature=1, top_p=0.9, max_new_tokens=2048),
        max_seq_len=7168,
        max_out_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
    ),
] #+ opensource_models
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
work_dir = "outputs/compassbench_v1_3_subjective_debug/"
