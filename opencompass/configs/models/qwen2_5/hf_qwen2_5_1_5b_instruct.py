from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='qwen2.5-1.5b-instruct-hf',
        path='Qwen/Qwen2.5-1.5B-Instruct',
        max_out_len=4096,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    )
]
