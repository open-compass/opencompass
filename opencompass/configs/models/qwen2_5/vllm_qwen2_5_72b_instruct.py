from opencompass.models import VLLMwithChatTemplate

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='qwen2_5-72b-instruct-vllm',
        path='Qwen/Qwen2.5-72B-Instruct',
        model_kwargs=dict(tensor_parallel_size=4),
        max_out_len=4096,
        batch_size=16,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=4),
    )
]
