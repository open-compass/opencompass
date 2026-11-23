from opencompass.models import OpenAISDKResponse

models = [
    dict(
        type=OpenAISDKResponse,
        abbr='grok-4.6-response',
        path='grok-4.6',
        key='XAI_API_KEY',
        openai_api_base='https://api.x.ai/v1',
        max_seq_len=500000,
        max_out_len=131072,
        query_per_second=1,
        max_workers=8,
        retry=10,
        temperature=None,
        openai_extra_kwargs=dict(
            reasoning=dict(effort='xhigh'),
            store=False,
        ),
        batch_size=8,
    )
]
