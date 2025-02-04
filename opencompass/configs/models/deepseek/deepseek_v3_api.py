from opencompass.models import OpenAISDK

models = [
    dict(
        abbr='deepseek_v3_api',
        type=OpenAISDK,
        path='deepseek-chat',
        key='ENV_DEEPSEEK',
        openai_api_base='https://api.deepseek.com/v1/',
        query_per_second=1,
        max_out_len=2048,
        max_seq_len=4096,
        batch_size=8,
        verbose=True,
    ),
]
