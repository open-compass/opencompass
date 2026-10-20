from opencompass.models import VLLM

settings = [
    ('qwen1.5-0.5b-vllm', 'Qwen/Qwen1.5-0.5B', 1),
    ('qwen1.5-1.8b-vllm', 'Qwen/Qwen1.5-1.8B', 1),
    ('qwen1.5-4b-vllm', 'Qwen/Qwen1.5-4B', 1),
    ('qwen1.5-7b-vllm', 'Qwen/Qwen1.5-7B', 1),
    ('qwen1.5-14b-vllm', 'Qwen/Qwen1.5-14B', 1),
    ('qwen1.5-32b-vllm', 'Qwen/Qwen1.5-32B', 2),
    ('qwen1.5-72b-vllm', 'Qwen/Qwen1.5-72B', 4),
    ('qwen1.5-110b-vllm', 'Qwen/Qwen1.5-110B', 4),
    ('qwen1.5-moe-a2.7b-vllm', 'Qwen/Qwen1.5-MoE-A2.7B', 1),
]

models = []
for abbr, path, num_gpus in settings:
    models.append(
        dict(
            type=VLLM,
            abbr=abbr,
            path=path,
            model_kwargs=dict(tensor_parallel_size=num_gpus),
            max_out_len=100,
            max_seq_len=2048,
            batch_size=32,
            generation_kwargs=dict(temperature=0),
            run_cfg=dict(num_gpus=num_gpus, num_procs=1),
        )
    )
