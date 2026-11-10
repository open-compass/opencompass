from opencompass.models import MiniMaxChatCompletionV2

models = [
    dict(
        type=MiniMaxChatCompletionV2,
        abbr='minimax-m3',
        path='MiniMax-M3',
        key='ENV',
        url='https://api.minimax.io/v1/chat/completions',
        max_seq_len=1000000,
        max_out_len=131072,
        query_per_second=1,
        retry=10,
        batch_size=8,
    )
]
