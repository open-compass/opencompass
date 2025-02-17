from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='deepseek-r1-distill-llama-8b-hf',
        path='deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
        max_out_len=4096,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    )
]
