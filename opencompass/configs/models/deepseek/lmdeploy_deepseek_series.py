from opencompass.models import LmdeployPytorchModel

settings = [
    ('deepseek-7b-base-hf', 'deepseek-ai/deepseek-llm-7b-base', 1),
    ('deepseek-67b-base-hf', 'deepseek-ai/deepseek-llm-67b-base', 4),
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
