from opencompass.models import HuggingFaceAboveV433Base


models = [
    dict(
        type=HuggingFaceAboveV433Base,
        abbr='internlm-7b-hf',
        path="internlm/internlm-7b",
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    )
]
