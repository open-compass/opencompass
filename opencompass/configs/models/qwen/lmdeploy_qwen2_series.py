from opencompass.models import TurboMindModel

settings = [
    # ('qwen2-0.5b-turbomind', 'Qwen/Qwen2-0.5B', 1),
    ('qwen2-1.5b-turbomind', 'Qwen/Qwen2-1.5B', 1),
    ('qwen2-7b-turbomind', 'Qwen/Qwen2-7B', 1),
    ('qwen2-72b-turbomind', 'Qwen/Qwen2-72B', 4),
    ('qwen2-57b-a14b-turbomind', 'Qwen/Qwen2-57B-A14B', 2),
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
