import torch
from opencompass.models import HuggingFaceBaseModel

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr='baichuan-m1-14b-base-hf',
        path='baichuan-inc/Baichuan-M1-14B-Base',
        max_out_len=1024,
        batch_size=8,
        model_kwargs=dict(device_map='auto', trust_remote_code=True, torch_dtype=torch.bfloat16),
        run_cfg=dict(num_gpus=1),
    )
]
