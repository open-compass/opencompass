import torch
from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='baichuan-m1-14b-instruct-hf',
        path='baichuan-inc/Baichuan-M1-14B-Instruct',
        max_out_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto', trust_remote_code=True, torch_dtype=torch.bfloat16),
        run_cfg=dict(num_gpus=1),
    )
]
