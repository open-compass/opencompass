from opencompass.models import VLLM


models = [
    dict(
        type=VLLM,
        abbr='qwen1.5-72b-vllm',
        path="Qwen/Qwen1.5-72B",
        model_kwargs=dict(tensor_parallel_size=4),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=32,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=4, num_procs=1),
    )
]
