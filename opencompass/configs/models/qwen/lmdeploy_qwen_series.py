from opencompass.models import TurboMindModel

settings = [
    ('qwen-1.8b-turbomind', 'Qwen/Qwen-1_8B', 1),
    ('qwen-7b-turbomind', 'Qwen/Qwen-7B', 1),
    ('qwen-14b-turbomind', 'Qwen/Qwen-14B', 1),
    ('qwen-72b-turbomind', 'Qwen/Qwen-72B', 4),
]

models = []
for abbr, path, num_gpus in settings:
    models.append(
        dict(
            type=TurboMindModel,
            abbr=abbr,
            path=path,
            engine_config=dict(session_len=7168, max_batch_size=16, tp=num_gpus),
            gen_config=dict(top_k=1, temperature=1, top_p=0.9, max_new_tokens=1024),
            max_out_len=1024,
            max_seq_len=7168,
            batch_size=16,
            concurrency=16,
            run_cfg=dict(num_gpus=num_gpus),
            stop_words=['<|im_end|>', '<|im_start|>'],
        )
    )
