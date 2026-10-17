from opencompass.models import OpenAISDK

api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
], )

models = [
    dict(
        abbr='atlascloud-qwen3.8-max',
        type=OpenAISDK,
        path='qwen/qwen3.8-max',
        key='ENV',
        key_env='ATLASCLOUD_API_KEY',
        meta_template=api_meta_template,
        query_per_second=1,
        openai_api_base='https://api.atlascloud.ai/v1',
        batch_size=1,
        max_out_len=2048,
        max_seq_len=32768,
        retry=3,
    ),
]
