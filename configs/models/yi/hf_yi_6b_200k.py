from opencompass.models import HuggingFaceAboveV433Base

models = [
    dict(
        type=HuggingFaceAboveV433Base,
        abbr='yi-6b-200k-hf',
        path='01-ai/Yi-6B-200K',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    )
]
