from os import getenv as gv
from opencompass.models import HuggingFaceCausalLM
from mmengine.config import read_base
with read_base():
    from .models.chatglm.hf_chatglm3_6b_32k import models as chatglm3_6b_32k_model
    from .models.yi.hf_yi_6b_chat import models as yi_6b_chat_model
    from .datasets.subjective.compassarena.compassarena_compare import subjective_datasets

from opencompass.models import HuggingFaceCausalLM, HuggingFace, HuggingFaceChatGLM3, OpenAI
from opencompass.models.openai_api import OpenAIAllesAPIN
from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.partitioners.sub_size import SubjectiveSizePartitioner
from opencompass.runners import LocalRunner
from opencompass.runners import SlurmSequentialRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask
from opencompass.summarizers import CompassArenaSummarizer

infer = dict(
    #partitioner=dict(type=NaivePartitioner),
    partitioner=dict(type=SizePartitioner, max_task_size=10000),
    runner=dict(
        type=SlurmSequentialRunner,
        partition='llm_dev2',
        quotatype='auto',
        max_num_workers=256,
        task=dict(type=OpenICLInferTask)),
)

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ]
)

gpt4 = dict(
        abbr='gpt4-turbo',
        type=OpenAI, path='gpt-4-1106-preview',
        key='',  # The key will be obtained from $OPENAI_API_KEY, but you can write down your key here as well
        meta_template=api_meta_template,
        query_per_second=1,
        max_out_len=2048,
        max_seq_len=4096,
        batch_size=4,
        retry=20,
        temperature = 1
)
models = [*chatglm3_6b_32k_model, *yi_6b_chat_model]
datasets = [*subjective_datasets]



work_dir = 'outputs/compass_arena/'

# -------------Inferen Stage ----------------------------------------

judge_model = dict(
        abbr='GPT4-Turbo',
        type=OpenAI, path='gpt-4-1106-preview',
        key='',  # The key will be obtained from $OPENAI_API_KEY, but you can write down your key here as well
        meta_template=api_meta_template,
        query_per_second=1,
        max_out_len=1024,
        max_seq_len=4096,
        batch_size=2,
        retry=20,
        temperature = 0
)
## ------------- Evaluation Configuration
eval = dict(
    partitioner=dict(
        type=SubjectiveSizePartitioner,
        strategy='split',
        max_task_size=10000,
        mode='m2n',
        base_models = [gpt4],
        compare_models = [*chatglm3_6b_32k_model, *yi_6b_chat_model, ]
    ),
    runner=dict(
        type=SlurmSequentialRunner,
        partition='llm_dev2',
        quotatype='auto',
        max_num_workers=32,
        task=dict(
            type=SubjectiveEvalTask,
            judge_cfg=judge_model
        )),
)


summarizer = dict(
    type=CompassArenaSummarizer
)