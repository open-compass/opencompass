from opencompass.models import MiniMaxChatCompletionV2

models = [
    dict(
        type=MiniMaxChatCompletionV2,
        abbr='minimax-m2.7',
        path='MiniMax-M2.7',
        key='MINIMAX_API_KEY',
        url='https://api.minimax.io/v1/text/chatcompletion_v2',
        max_seq_len=204800,
        max_out_len=4096,
        query_per_second=1,
        retry=10,
        batch_size=8,
        max_workers=8,
    )
]
