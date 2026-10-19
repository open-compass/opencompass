from opencompass.models import VLLMwithChatTemplate

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='gemma-7b-it-vllm',
        path='google/gemma-7b-it',
        model_kwargs=dict(tensor_parallel_size=1, gpu_memory_utilization=0.5),
        max_out_len=1024,
        batch_size=16,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=1),
    )
]
