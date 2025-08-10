from opencompass.models import VLLM

models = [
    dict(
        type=VLLM,
        abbr='mistral-7b-v0.2-vllm',
        path='mistral-community/Mistral-7B-v0.2',
        model_kwargs=dict(dtype='bfloat16', tensor_parallel_size=1),
        max_out_len=1024,
        max_seq_len=8192,
        batch_size=16,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=1),
    )
]
