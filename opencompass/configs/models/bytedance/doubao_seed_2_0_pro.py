from opencompass.models import OpenAISDKStreaming

models = [
    dict(
        type=OpenAISDKStreaming,
        abbr='doubao-seed-2.0-pro',
        path='doubao-seed-2-0-pro-260215',
        key='ARK_API_KEY',
        openai_api_base='https://ark.cn-beijing.volces.com/api/v3',
        max_seq_len=262144,
        max_out_len=131072,
        query_per_second=1,
        max_workers=8,
        retry=10,
        temperature=0.6,
        extra_body=dict(thinking=dict(type='enabled')),
        batch_size=8,
    )
]
