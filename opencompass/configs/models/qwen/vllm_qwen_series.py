from opencompass.models import VLLM

settings = [
    ('qwen-1.8b-vllm', 'Qwen/Qwen-1_8B', 1),
    ('qwen-7b-vllm', 'Qwen/Qwen-7B', 1),
    ('qwen-14b-vllm', 'Qwen/Qwen-14B', 1),
    ('qwen-72b-vllm', 'Qwen/Qwen-72B', 4),
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
