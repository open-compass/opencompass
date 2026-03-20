from opencompass.models import MiniMaxAPI

api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
], )

models = [
    dict(
        abbr='MiniMax-M2.5-highspeed',
        type=MiniMaxAPI,
        path='MiniMax-M2.5-highspeed',
        key='ENV',
        meta_template=api_meta_template,
        query_per_second=2,
        max_out_len=4096,
        max_seq_len=204800,
        batch_size=8,
    ),
]
