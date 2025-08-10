from opencompass.models import VLLM

settings = [
    ('internlm2-1.8b-vllm', 'internlm/internlm2-1_8b', 1),
    ('internlm2-7b-vllm', 'internlm/internlm2-7b', 1),
    ('internlm2-base-7b-vllm', 'internlm/internlm2-base-7b', 1),
    ('internlm2-20b-vllm', 'internlm/internlm2-20b', 2),
    ('internlm2-base-20b-vllm', 'internlm/internlm2-base-20b', 2),
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
