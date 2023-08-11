from mmengine.config import read_base
from opencompass.models import OpenAI
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    from .datasets.collections.chat_medium import datasets
    from .summarizers.medium import summarizer

# GPT4 needs a special humaneval postprocessor
from opencompass.datasets.humaneval import humaneval_gpt_postprocess
for _dataset in datasets:
    if _dataset['path'] == 'openai_humaneval':
        _dataset['eval_cfg']['pred_postprocessor']['type'] = humaneval_gpt_postprocess


api_meta_template = dict(
    round=[
            dict(role='HUMAN', api_role='HUMAN'),
            dict(role='BOT', api_role='BOT', generate=True),
    ],
)

models = [
    dict(abbr='GPT4',
        type=OpenAI, path='gpt-4-0613',
        key='ENV',  # The key will be obtained from $OPENAI_API_KEY, but you can write down your key here as well
        meta_template=api_meta_template,
        query_per_second=1,
        max_out_len=2048, max_seq_len=2048, batch_size=8),
]

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=4,
        task=dict(type=OpenICLInferTask)),
)
