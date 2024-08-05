
from mmengine.config import read_base
from opencompass.models import OpenAI
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask


SWIFT_DEPLOY_API = 'http://127.0.0.1:8000/v1/chat/completions'

# TODO: BY JASON ONLY FOR TEST!
with read_base():
    from .datasets.ceval.ceval_gen import ceval_datasets

    from .summarizers.teval import summarizer


datasets = [*ceval_datasets]


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
    dict(abbr='LlaMA-3-8B-INSTRUCT',
         type=OpenAI,
         path='llama3-8b-instruct',
         # The key will be obtained from $OPENAI_API_KEY, but you can write down your key here as well
         key='your_openai_api_key',     # No need for swift deployment API
         meta_template=api_meta_template,
         query_per_second=1,
         max_out_len=2048,
         max_seq_len=2048,
         batch_size=1,
         generation_kwargs=dict(temperature=0.7),
         run_cfg=dict(num_gpus=0),
         openai_api_base=SWIFT_DEPLOY_API,
         ),
]


infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=4,
        task=dict(type=OpenICLInferTask)),
)
