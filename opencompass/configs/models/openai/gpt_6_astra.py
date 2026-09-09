from opencompass.models import OpenAISDKResponse

models = [
    dict(
        type=OpenAISDKResponse,
        abbr='gpt-6-astra-response',
        path='gpt-6-astra',
        key='ENV',  # OPENAI_API_KEY
        openai_api_base='https://api.openai.com/v1',
        max_seq_len=1000000,
        max_out_len=131072,
        query_per_second=1,
        max_workers=8,
        retry=10,
        openai_extra_kwargs=dict(
            reasoning=dict(effort='max'),
        ),
        batch_size=8,
    )
]
