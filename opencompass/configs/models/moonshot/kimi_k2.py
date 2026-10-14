from opencompass.models import OpenAISDK

api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
], )

models = [
    dict(
        abbr='kimi-k2-0711-preview',
        type=OpenAISDK,
        path='kimi-k2-0711-preview',
        key='your-api-key-here',  # Set your API key here
        meta_template=api_meta_template,
        query_per_second=1,
        openai_api_base='https://api.moonshot.cn/v1',
        batch_size=1,
        temperature=1,
        max_seq_len=131072,
        retry=10,
        ),
]
