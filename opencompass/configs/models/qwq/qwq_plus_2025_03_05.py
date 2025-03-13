from opencompass.models import Qwen

api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
], )

models = [
    dict(
        abbr='QwQ-Plus-2025-03-05',
        type=Qwen,
        path='qwq-plus-2025-03-05',
        key='ENV',  # The key will be obtained from $OPENAI_API_KEY, but you can write down your key here as well
        stream=True,
        meta_template=api_meta_template,
        query_per_second=1,
        max_out_len=8192,
        max_seq_len=8192,
        batch_size=8),
]
