from opencompass.models import Fireworks
from opencompass.partitioners import NaivePartitioner
from opencompass.runners.local_api import LocalAPIRunner
from opencompass.tasks import OpenICLInferTask

api_meta_template = dict(
    round=[
            dict(role='HUMAN', api_role='HUMAN'),
            dict(role='BOT', api_role='BOT', generate=True),
    ],
)

mistral_7b = [
    dict(abbr='mistral-7b',
        type=Fireworks, path='accounts/fireworks/models/mistral-7b',
        key='ENV',  # The key will be obtained from $FIREWORKS_API_KEY, but you can write down your key here as well
        meta_template=api_meta_template,
        query_per_second=1,
        max_out_len=2048,
        max_seq_len=4096,
        batch_size=8),
]

