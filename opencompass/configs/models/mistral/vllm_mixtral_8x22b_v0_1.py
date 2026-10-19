from opencompass.models import VLLM

models = [
    dict(
        type=VLLM,
        abbr='mixtral-8x22b-v0.1-vllm',
        path='mistralai/Mixtral-8x22B-v0.1',
        model_kwargs=dict(dtype='bfloat16', tensor_parallel_size=8),
        max_out_len=1024,
        max_seq_len=8192,
        batch_size=16,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=8),
    )
]
