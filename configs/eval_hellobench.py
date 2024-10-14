from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.subjective.hellobench.hellobench import hellobench_datasets
from opencompass.models import HuggingFacewithChatTemplate
from opencompass.partitioners import NaivePartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask

# -------------Inference Stage ----------------------------------------
# For subjective evaluation, we often set do sample for models
models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='glm-4-9b-chat-hf',
        path='THUDM/glm-4-9b-chat',
        max_out_len=16384,
        generation_kwargs=dict(
            temperature=0.8,
            do_sample=True, #For subjective evaluation, we suggest you do set do_sample when running model inference!
        ),
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
        ),
        batch_size=1,
        run_cfg=dict(num_gpus=2, num_procs=1),
        stop_words=['<|endoftext|>', '<|user|>', '<|observation|>'],
    )
]

datasets = [*hellobench_datasets] # add datasets you want

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner, max_num_workers=16, task=dict(type=OpenICLInferTask)),
)
# -------------Evalation Stage ----------------------------------------

# ------------- JudgeLLM Configuration
judge_models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='glm-4-9b-chat-hf',
        path='THUDM/glm-4-9b-chat',
        max_out_len=16384,
        generation_kwargs=dict(
            temperature=0.8,
            do_sample=True, #For subjective evaluation, we suggest you do set do_sample when running model inference!
        ),
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
        ),
        batch_size=1,
        run_cfg=dict(num_gpus=2, num_procs=1),
        stop_words=['<|endoftext|>', '<|user|>', '<|observation|>'],
    )
]

## ------------- Evaluation Configuration
eval = dict(
    partitioner=dict(type=SubjectiveNaivePartitioner, models=models, judge_models=judge_models,),
    runner=dict(type=LocalRunner, max_num_workers=16, task=dict(type=SubjectiveEvalTask)),
)

work_dir = 'outputs/hellobench/'
