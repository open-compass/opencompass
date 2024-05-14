from mmengine.config import read_base

with read_base():
    from .datasets.subjective.alpaca_eval.alpacav1_judgeby_gpt4 import subjective_datasets as alpacav1
    from .datasets.subjective.alpaca_eval.alpacav2_judgeby_gpt4 import subjective_datasets as alpacav2

from opencompass.models import HuggingFaceCausalLM, HuggingFace, HuggingFaceChatGLM3
from opencompass.models.openai_api import OpenAI, OpenAIAllesAPIN
from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.partitioners.sub_size import SubjectiveSizePartitioner
from opencompass.runners import LocalRunner
from opencompass.runners import SlurmSequentialRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask
from opencompass.summarizers import AlpacaSummarizer

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
    reserved_roles=[dict(role='SYSTEM', api_role='SYSTEM')],
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
        batch_size=8,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]

datasets = [*alpacav2]

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

## ------------- Evaluation Configuration
eval = dict(
    partitioner=dict(
        type=SubjectiveSizePartitioner, max_task_size=1000, mode='m2n', base_models=[gpt4], compare_models=models,
        infer_order='random',
        judge_models=judge_models
    ),
    runner=dict(type=LocalRunner, max_num_workers=2, task=dict(type=SubjectiveEvalTask)),
    given_pred = [{'abbr':'gpt4-turbo', 'path':''}]
)
work_dir = 'outputs/alpaca/'



summarizer = dict(type=AlpacaSummarizer, judge_type='v2')
