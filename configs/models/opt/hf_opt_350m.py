from opencompass.models import HuggingFaceBaseModel

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr='opt-350m-hf',
        path='facebook/opt-350m',
        max_out_len=1024,
        batch_size=32,
        run_cfg=dict(num_gpus=1),
    )
]
