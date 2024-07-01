from opencompass.models import VLLMwithChatTemplate

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='qwen2-1.5b-instruct-vllm',
        path='Qwen/Qwen2-1.5B-Instruct',
        model_kwargs=dict(tensor_parallel_size=1),
        max_out_len=1024,
        batch_size=16,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=1),
    )
]
