from opencompass.models import VLLM


models = [
    dict(
        type=VLLM,
        abbr='swe-llama-7b-vllm',
        path='princeton-nlp/SWE-Llama-7b',
        model_kwargs=dict(
            tensor_parallel_size=1,
            max_model_len=8192,
            gpu_memory_utilization=0.7,
            trust_remote_code=True,
        ),
        generation_kwargs=dict(
            temperature=0.0,
            top_p=0.95,
        ),
        max_out_len=4096,
        batch_size=2,
        run_cfg=dict(num_gpus=1),
    )
]
