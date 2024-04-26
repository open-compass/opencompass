from opencompass.models import HuggingFaceBaseModel

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr='falcon-40b-hf',
        path='tiiuae/falcon-40b',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=4),
    )
]
