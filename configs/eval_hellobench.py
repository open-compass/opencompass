from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.subjective.hellobench.hellobench import hellobench_datasets
from opencompass.models import HuggingFacewithChatTemplate, OpenAI
from opencompass.partitioners import NaivePartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask
from opencompass.summarizers import DefaultSubjectiveSummarizer


api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ]
)

# -------------Inference Stage ----------------------------------------
# For subjective evaluation, we often set do sample for models
# make sure your models' generation parameters are set properly, for example, if you set temperature=0.8, make sure you set all models' temperature to 0.8
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
# we recommand to use gpt4o-mini as the judge model

# if you want to use open-source LLMs as judge models, you can uncomment the following code
# judge_models = [
#     dict(
#         type=HuggingFacewithChatTemplate,
#         abbr='glm-4-9b-chat-hf',
#         path='THUDM/glm-4-9b-chat',
#         max_out_len=16384,
#         generation_kwargs=dict(
#             temperature=0.8,
#             do_sample=True, #For subjective evaluation, we suggest you do set do_sample when running model inference!
#         ),
#         model_kwargs=dict(
#             device_map='auto',
#             trust_remote_code=True,
#         ),
#         batch_size=1,
#         run_cfg=dict(num_gpus=2, num_procs=1),
#         stop_words=['<|endoftext|>', '<|user|>', '<|observation|>'],
#     )
# ]

judge_models = [dict(
    abbr='GPT4o',
    type=OpenAI,
    path='gpt-4o',
    key='xxxx',  # The key will be obtained from $OPENAI_API_KEY, but you can write down your key here as well
    meta_template=api_meta_template,
    query_per_second=16,
    max_out_len=4096,
    batch_size=1,
    temperature=0.8,
    seed=42,
)]

## ------------- Evaluation Configuration
eval = dict(
    partitioner=dict(type=SubjectiveNaivePartitioner, models=models, judge_models=judge_models,),
    runner=dict(type=LocalRunner, max_num_workers=16, task=dict(type=SubjectiveEvalTask)),
)

summarizer = dict(type=DefaultSubjectiveSummarizer)
work_dir = 'outputs/hellobench/'
