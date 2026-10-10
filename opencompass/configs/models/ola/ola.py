from opencompass.models import OlaModel
models = [
    dict(
        type=OlaModel, 
        path="THUdyh/Ola-7b",
        max_seq_len=2048,
        abbr='ola', 
        max_out_len=1024,
        batch_size=1,
        run_cfg=dict(num_gpus=1), 
    )
]
