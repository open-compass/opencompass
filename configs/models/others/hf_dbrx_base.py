from opencompass.models import HuggingFaceAboveV433Base

models = [
    dict(
        type=HuggingFaceAboveV433Base,
        abbr='dbrx-base-hf',
        path='databricks/dbrx-base',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=8),
    )
]
