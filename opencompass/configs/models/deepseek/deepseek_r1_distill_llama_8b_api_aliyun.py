from opencompass.models import OpenAISDK

models = [
    dict(
        abbr='deepseek_r1_distill_llama_8b_api_aliyun',
        type=OpenAISDK,
        path='deepseek-r1-distill-llama-8b',
        key='ENV_ALIYUN',
        openai_api_base='https://dashscope.aliyuncs.com/compatible-mode/v1',
        query_per_second=1,
        max_out_len=2048,
        max_seq_len=4096,
        batch_size=8,
        retry=30,
        verbose=True,
    ),
]
