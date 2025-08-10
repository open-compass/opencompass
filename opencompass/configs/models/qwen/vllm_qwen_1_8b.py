from opencompass.models import VLLM

models = [
    dict(
        type=VLLM,
        abbr='qwen-1.8b-vllm',
        path='Qwen/Qwen-1_8B',
        model_kwargs=dict(tensor_parallel_size=1),
        max_out_len=1024,
        max_seq_len=8192,
        batch_size=16,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=1),
    )
]
