from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='mixtral-8x22b-instruct-v0.1-hf',
        path='mistralai/Mixtral-8x22B-Instruct-v0.1',
        max_out_len=1024,
        batch_size=4,
        run_cfg=dict(num_gpus=8),
    )
]
