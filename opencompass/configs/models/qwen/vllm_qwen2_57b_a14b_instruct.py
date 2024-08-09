from opencompass.models import VLLMwithChatTemplate

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='qwen2-57b-a14b-instruct-vllm',
        path='Qwen/Qwen2-57B-A14B-Instruct',
        model_kwargs=dict(tensor_parallel_size=2),
        max_out_len=1024,
        batch_size=16,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=2),
    )
]
