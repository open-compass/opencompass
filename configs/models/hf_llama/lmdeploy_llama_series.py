from opencompass.models import TurboMindModel

settings = [
    ('llama-7b-turbomind', 'huggyllama/llama-7b', 1),
    ('llama-13b-turbomind', 'huggyllama/llama-13b', 1),
    ('llama-30b-turbomind', 'huggyllama/llama-30b', 2),
    ('llama-65b-turbomind', 'huggyllama/llama-65b', 4),
    ('llama-2-7b-turbomind', 'meta-llama/Llama-2-7b-hf', 1),
    ('llama-2-13b-turbomind', 'meta-llama/Llama-2-13b-hf', 1),
    ('llama-2-70b-turbomind', 'meta-llama/Llama-2-70b-hf', 4),
    ('llama-3-8b-turbomind', 'meta-llama/Meta-Llama-3-8B', 1),
    ('llama-3-70b-turbomind', 'meta-llama/Meta-Llama-3-70B', 4),
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
