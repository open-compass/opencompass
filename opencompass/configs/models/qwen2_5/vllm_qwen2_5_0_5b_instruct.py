from opencompass.models import VLLMwithChatTemplate

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='qwen2.5-0.5b-instruct-vllm',
        path='Qwen/Qwen2.5-0.5B-Instruct',
        model_kwargs=dict(tensor_parallel_size=1, gpu_memory_utilization=0.5),
        max_out_len=4096,
        batch_size=16,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=1),
    )
]
