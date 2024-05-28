from opencompass.models import LmdeployPytorchModel

settings = [
    ('qwen1.5-0.5b-pytorch', 'Qwen/Qwen1.5-0.5B', 1),
    ('qwen1.5-1.8b-pytorch', 'Qwen/Qwen1.5-1.8B', 1),
    ('qwen1.5-4b-pytorch', 'Qwen/Qwen1.5-4B', 1),
    ('qwen1.5-7b-pytorch', 'Qwen/Qwen1.5-7B', 1),
    ('qwen1.5-14b-pytorch', 'Qwen/Qwen1.5-14B', 1),
    ('qwen1.5-32b-pytorch', 'Qwen/Qwen1.5-32B', 2),
    ('qwen1.5-72b-pytorch', 'Qwen/Qwen1.5-72B', 4),
    ('qwen1.5-110b-pytorch', 'Qwen/Qwen1.5-110B', 4),
    ('qwen1.5-moe-a2.7b-pytorch', 'Qwen/Qwen1.5-MoE-A2.7B', 1),
]

models = []
for abbr, path, num_gpus in settings:
    models.append(
        dict(
            type=LmdeployPytorchModel,
            abbr=abbr,
            path=path,
            engine_config=dict(session_len=2048, max_batch_size=16, tp=num_gpus),
            gen_config=dict(top_k=1, temperature=1, top_p=0.9, max_new_tokens=1024),
            max_out_len=1024,
            max_seq_len=2048,
            batch_size=16,
            concurrency=16,
            run_cfg=dict(num_gpus=num_gpus),
        )
    )
