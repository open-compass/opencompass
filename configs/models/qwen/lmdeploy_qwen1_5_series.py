from opencompass.models import TurboMindModel

settings = [
    # ('qwen1.5-0.5b-turbomind', 'Qwen/Qwen1.5-0.5B', 1),
    ('qwen1.5-1.8b-turbomind', 'Qwen/Qwen1.5-1.8B', 1),
    ('qwen1.5-4b-turbomind', 'Qwen/Qwen1.5-4B', 1),
    ('qwen1.5-7b-turbomind', 'Qwen/Qwen1.5-7B', 1),
    ('qwen1.5-14b-turbomind', 'Qwen/Qwen1.5-14B', 1),
    ('qwen1.5-32b-turbomind', 'Qwen/Qwen1.5-32B', 2),
    ('qwen1.5-72b-turbomind', 'Qwen/Qwen1.5-72B', 4),
    ('qwen1.5-110b-turbomind', 'Qwen/Qwen1.5-110B', 4),
    ('qwen1.5-moe-a2.7b-turbomind', 'Qwen/Qwen1.5-MoE-A2.7B', 1),
]

models = []
for abbr, path, num_gpus in settings:
    models.append(
        dict(
            type=TurboMindModel,
            abbr=abbr,
            path=path,
            engine_config=dict(session_len=7168, max_batch_size=8, tp=num_gpus),
            gen_config=dict(top_k=1, top_p=0.8, temperature=1.0, max_new_tokens=1024),
            max_out_len=1024,
            max_seq_len=7168,
            batch_size=8,
            concurrency=8,
            run_cfg=dict(num_gpus=num_gpus, num_procs=1),
        )
    )
