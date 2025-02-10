from opencompass.models import OpenAISDK

models = [
    dict(
        abbr='deepseek_r1_api_siliconflow',
        type=OpenAISDK,
        path='Pro/deepseek-ai/DeepSeek-R1',
        key='ENV_SILICONFLOW',
        openai_api_base='https://api.siliconflow.cn/v1/',
        query_per_second=0.1,
        max_completion_tokens=8192,
        batch_size=1,
        verbose=True,
    ),
]
