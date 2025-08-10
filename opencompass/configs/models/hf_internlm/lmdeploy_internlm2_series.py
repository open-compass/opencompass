from opencompass.models import TurboMindModel

settings = [
    ('internlm2-1.8b-turbomind', 'internlm/internlm2-1_8b', 1),
    ('internlm2-7b-turbomind', 'internlm/internlm2-7b', 1),
    ('internlm2-base-7b-turbomind', 'internlm/internlm2-base-7b', 1),
    ('internlm2-20b-turbomind', 'internlm/internlm2-20b', 2),
    ('internlm2-base-20b-turbomind', 'internlm/internlm2-base-20b', 2),
]

models = []
for abbr, path, num_gpus in settings:
    models.append(
        dict(
            type=TurboMindModel,
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
