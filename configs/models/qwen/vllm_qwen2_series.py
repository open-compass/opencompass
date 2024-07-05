from opencompass.models import VLLM

settings = [
    ('qwen2-0.5b-vllm', 'Qwen/Qwen2-0.5B', 1),
    ('qwen2-1.5b-vllm', 'Qwen/Qwen2-1.5B', 1),
    ('qwen2-7b-vllm', 'Qwen/Qwen2-7B', 1),
    ('qwen2-72b-vllm', 'Qwen/Qwen2-72B', 4),
    ('qwen2-57b-a14b-vllm', 'Qwen/Qwen2-57B-A14B', 2),
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
