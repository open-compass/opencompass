from opencompass.models import VLLM

settings = [
    ('llama-7b-vllm', 'huggyllama/llama-7b', 1),
    ('llama-13b-vllm', 'huggyllama/llama-13b', 1),
    ('llama-30b-vllm', 'huggyllama/llama-30b', 2),
    ('llama-65b-vllm', 'huggyllama/llama-65b', 4),
    ('llama-2-7b-vllm', 'meta-llama/Llama-2-7b-hf', 1),
    ('llama-2-13b-vllm', 'meta-llama/Llama-2-13b-hf', 1),
    ('llama-2-70b-vllm', 'meta-llama/Llama-2-70b-hf', 4),
    ('llama-3-8b-vllm', 'meta-llama/Meta-Llama-3-8B', 1),
    ('llama-3-70b-vllm', 'meta-llama/Meta-Llama-3-70B', 4),
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
