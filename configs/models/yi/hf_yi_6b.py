from opencompass.models import HuggingFaceBaseModel

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr='yi-6b-hf',
        path='01-ai/Yi-6B',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    )
]
