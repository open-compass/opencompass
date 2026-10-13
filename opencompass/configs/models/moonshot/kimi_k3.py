from opencompass.models import OpenAISDKStreaming

models = [
    dict(
        type=OpenAISDKStreaming,
        abbr='kimi-k3',
        path='kimi-k3',
        key='MOONSHOT_API_KEY',
        openai_api_base='https://api.moonshot.cn/v1',
        max_seq_len=262144,
        max_out_len=131072,
        query_per_second=1,
        max_workers=8,
        retry=10,
        temperature=0.6,
        batch_size=8,
    )
]
