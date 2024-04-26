from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='mixtral-8x7b-instruct-v0.1-hf',
        path='mistralai/Mixtral-8x7B-Instruct-v0.1',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=4),
    )
]
