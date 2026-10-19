from opencompass.models import OpenAISDKStreaming

models = [
    dict(
        type=OpenAISDKStreaming,
        abbr='glm-5.3-flash-api',
        path='glm-5.3-flash',
        key='ZAI_API_KEY',
        openai_api_base='https://api.z.ai/api/paas/v4',
        max_seq_len=1000000,
        max_out_len=131072,
        query_per_second=1,
        max_workers=8,
        retry=10,
        temperature=0.6,
        batch_size=8,
    )
]
