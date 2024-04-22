from opencompass.models import HuggingFaceAboveV433Chat

models = [
    dict(
        type=HuggingFaceAboveV433Chat,
        abbr='mixtral-8x22b-instruct-v0.1',
        path='mistralai/Mixtral-8x22B-Instruct-v0.1',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=4),
    )
]
