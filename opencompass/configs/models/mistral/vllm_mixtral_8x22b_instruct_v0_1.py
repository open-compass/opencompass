from opencompass.models import VLLMwithChatTemplate


models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='mixtral-8x22b-instruct-v0.1-vllm',
        path='mistralai/Mixtral-8x22B-Instruct-v0.1',
        model_kwargs=dict(tensor_parallel_size=8),
        max_out_len=256,
        batch_size=16,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=8),
    )
]
