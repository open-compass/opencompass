from opencompass.models import TeleChat
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

api_meta_template = dict(
    round=[
            dict(role='HUMAN', api_role='HUMAN'),
            dict(role='BOT', api_role='BOT', generate=True),
    ],
)

models = [
    dict(
        type=TeleChat,
        path='TeleChat-thinking',
        key='ENV',
        meta_template=api_meta_template,
        query_per_second=1,
        retry=5,
        max_out_len=28672,
        max_seq_len=32768,
        batch_size=8,
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]
