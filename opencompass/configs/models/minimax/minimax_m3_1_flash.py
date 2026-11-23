from opencompass.models import MiniMaxChatCompletionV2

models = [
    dict(
        type=MiniMaxChatCompletionV2,
        abbr='minimax-m3.1-flash',
        path='MiniMax-M3.1-Flash',
        key='ENV',
        url='https://api.minimax.io/v1/chat/completions',
        max_seq_len=1000000,
        max_out_len=524288,
        query_per_second=1,
        retry=10,
        batch_size=8,
    )
]
