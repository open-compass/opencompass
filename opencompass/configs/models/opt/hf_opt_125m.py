from opencompass.models import HuggingFaceBaseModel

opt125m = dict(
    type=HuggingFaceBaseModel,
    abbr='opt-125m-hf',
    path='facebook/opt-125m',
    max_out_len=1024,
    batch_size=64,
    run_cfg=dict(num_gpus=1),
)

models = [opt125m]
