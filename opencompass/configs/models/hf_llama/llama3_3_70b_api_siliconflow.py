from opencompass.models import OpenAISDK

models = [
    dict(
        abbr='llama3_3_70b_api_siliconflow',
        type=OpenAISDK,
        path='meta-llama/Llama-3.3-70B-Instruct',
        key='ENV_SILICONFLOW',
        openai_api_base='https://api.siliconflow.cn/v1/',
        query_per_second=1,
        max_out_len=4096,
        max_seq_len=4096,
        batch_size=1,
        retry=30,
        verbose=True,
    ),
]
