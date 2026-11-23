from opencompass.models import OpenAISDKStreaming

models = [
    dict(
        type=OpenAISDKStreaming,
        abbr='deepseek-v4-flash',
        path='deepseek-v4-flash',
        key='DEEPSEEK_API_KEY',
        openai_api_base='https://api.deepseek.com',
        max_seq_len=1000000,
        max_out_len=131072,
        query_per_second=1,
        max_workers=8,
        retry=10,
        temperature=None,
        openai_extra_kwargs=dict(reasoning_effort='max'),
        batch_size=8,
    )
]
