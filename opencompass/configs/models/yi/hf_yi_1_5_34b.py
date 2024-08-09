from opencompass.models import HuggingFaceBaseModel

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr='yi-1.5-34b-hf',
        path='01-ai/Yi-1.5-34B',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=2),
    )
]
