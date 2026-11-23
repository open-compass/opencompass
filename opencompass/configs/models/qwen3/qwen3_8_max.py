from opencompass.models import OpenAISDKStreaming

models = [
    dict(
        type=OpenAISDKStreaming,
        abbr='qwen3.8-max-api',
        path='qwen3.8-max',
        key='DASHSCOPE_API_KEY',
        openai_api_base='https://dashscope.aliyuncs.com/compatible-mode/v1',
        max_seq_len=1000000,
        max_out_len=131072,
        query_per_second=1,
        max_workers=8,
        retry=10,
        temperature=0.6,
        openai_extra_kwargs=dict(enable_thinking=True, top_p=0.95),
        batch_size=8,
    )
]
