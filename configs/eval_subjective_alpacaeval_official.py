from mmengine.config import read_base

with read_base():
    from .datasets.subjective.alpaca_eval.alpacav2_judgeby_gpt4 import subjective_datasets as alpacav2

from opencompass.models import HuggingFaceCausalLM, HuggingFace, HuggingFaceChatGLM3
from opencompass.models.openai_api import OpenAI
from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.partitioners.sub_size import SubjectiveSizePartitioner
from opencompass.runners import LocalRunner
from opencompass.runners import SlurmSequentialRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.tasks.outer_eval.alpacaeval import AlpacaEvalTask
from opencompass.summarizers import AlpacaSummarizer

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
    reserved_roles=[dict(role='SYSTEM', api_role='SYSTEM')],
)


# To run this config, please ensure to successfully installed `alpaca-eval==0.6` and `scikit-learn==1.5`

# -------------Inference Stage ----------------------------------------

# For subjective evaluation, we often set do sample for models
models = [
    dict(
        type=HuggingFaceChatGLM3,
        abbr='chatglm3-6b',
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

datasets = [*alpacav2]

# -------------Evalation Stage ----------------------------------------

## ------------- JudgeLLM Configuration
gpt4_judge = dict(
    abbr='GPT4-Turbo',
    path='gpt-4-1106-preview',
    key='',  # The key will be obtained from $OPENAI_API_KEY, but you can write down your key here as well
    config='weighted_alpaca_eval_gpt4_turbo'
)
## ------------- Evaluation Configuration
eval = dict(
    partitioner=dict(
        type=NaivePartitioner
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=256,
        task=dict(type=AlpacaEvalTask, judge_cfg=gpt4_judge),
    )
)
work_dir = 'outputs/alpaca/'
