from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='deepseek-r1-distill-llama-70b-hf',
        path='deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
        max_out_len=4096,
        batch_size=8,
        run_cfg=dict(num_gpus=8),
    )
]
