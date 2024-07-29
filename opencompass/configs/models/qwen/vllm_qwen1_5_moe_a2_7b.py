from opencompass.models import VLLM

models = [
    dict(
        type=VLLM,
        abbr='qwen1.5-moe-a2.7b-vllm',
        path='Qwen/Qwen1.5-MoE-A2.7B',
        model_kwargs=dict(tensor_parallel_size=1, gpu_memory_utilization=0.5),
        max_out_len=1024,
        max_seq_len=8192,
        batch_size=16,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=1),
    )
]
