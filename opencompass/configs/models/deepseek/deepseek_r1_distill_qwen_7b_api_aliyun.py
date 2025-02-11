from opencompass.models import OpenAISDK

models = [
    dict(
        abbr='deepseek_r1_distill_qwen_7b_api_aliyun',
        type=OpenAISDK,
        path='deepseek-r1-distill-qwen-7b',
        key='ENV_ALIYUN',
        openai_api_base='https://dashscope.aliyuncs.com/compatible-mode/v1',
        query_per_second=1,
        max_out_len=8192,
        max_seq_len=32768,
        batch_size=1,
        retry=30,
        verbose=True,
    ),
]
