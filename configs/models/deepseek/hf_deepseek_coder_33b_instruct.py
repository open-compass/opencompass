from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='deepseek-coder-33b-hf',
        path='deepseek-ai/deepseek-coder-33b-instruct',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=2),
    )
]
