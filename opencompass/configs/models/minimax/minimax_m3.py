from opencompass.models import MiniMaxAPI

api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
], )

models = [
    dict(
        abbr='MiniMax-M3',
        type=MiniMaxAPI,
        path='MiniMax-M3',
        key='ENV',
        meta_template=api_meta_template,
        query_per_second=2,
        max_out_len=4096,
        max_seq_len=524288,
        batch_size=8,
    ),
]
