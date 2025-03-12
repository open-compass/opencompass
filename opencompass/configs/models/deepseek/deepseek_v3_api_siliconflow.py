from opencompass.models import OpenAISDK

models = [
    dict(
        abbr='deepseek_v3_api_siliconflow',
        type=OpenAISDK,
        path='Pro/deepseek-ai/DeepSeek-V3',
        key='ENV_SILICONFLOW',
        openai_api_base='https://api.siliconflow.cn/v1/',
        query_per_second=0.1,
        max_out_len=8192,
        max_seq_len=8192,
        batch_size=8,
        verbose=True,
        retry=5,
    ),
]
