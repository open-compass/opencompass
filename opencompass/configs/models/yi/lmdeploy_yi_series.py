from opencompass.models import LmdeployPytorchModel

settings = [
    ('yi-6b-pytorch', '01-ai/Yi-6B', 1),
    ('yi-34b-pytorch', '01-ai/Yi-34B', 2),
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
