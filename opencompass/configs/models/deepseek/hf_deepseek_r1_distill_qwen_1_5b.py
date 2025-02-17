from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='deepseek-r1-distill-qwen-1.5b-hf',
        path='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
        max_out_len=4096,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    )
]
