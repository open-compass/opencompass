from opencompass.models import OpenAISDK

models = [
    dict(
        abbr='deepseek_r1_api_siliconflow',
        type=OpenAISDK,
        path='Pro/deepseek-ai/DeepSeek-R1',
        key='ENV_SILICONFLOW',
        openai_api_base='https://api.siliconflow.cn/v1/',
        query_per_second=0.1,
        max_out_len=8192,
        max_seq_len=32768,
        batch_size=1,
        retry=30,
        verbose=True,
    ),
]
