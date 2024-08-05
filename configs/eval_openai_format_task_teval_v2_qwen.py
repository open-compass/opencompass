
from mmengine.config import read_base
from opencompass.models import OpenAI
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask


# For swift model serving
SWIFT_DEPLOY_URL = 'http://127.0.0.1:8000/v1/chat/completions'


# TODO: BY JASON ONLY FOR TEST!
with read_base():
    from .datasets.teval_v2.teval_v2_en_gen_1ac254 import teval_datasets as teval_en_datasets
    from .datasets.teval_v2.teval_v2_zh_gen_1ac254 import teval_datasets as teval_zh_datasets

    from .summarizers.teval_v2 import summarizer

# datasets = [*teval_en_datasets, *teval_zh_datasets]
datasets = [*teval_en_datasets]     # TODO： BY JASON ONLY FOR TEST!


api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True)
    ],
    reserved_roles=[
        dict(role='SYSTEM', api_role='SYSTEM'),
    ],
)


models = [
    dict(abbr='Qwen-7B-Chat',
         type=OpenAI,
         path='qwen-7b-chat',
         # The key will be obtained from $OPENAI_API_KEY, but you can write down your key here as well
         key='your_openai_api_key',  # No need for swift deployment API
         meta_template=api_meta_template,
         query_per_second=1,
         max_out_len=2048,
         max_seq_len=4096,
         batch_size=1,
         run_cfg=dict(num_gpus=0),
         openai_api_base=SWIFT_DEPLOY_URL,
         ),
]


infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=4,
        task=dict(type=OpenICLInferTask)),
)
