from opencompass.models import VLLMwithChatTemplate


models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='mistral-7b-instruct-v0.2-vllm',
        path='mistralai/Mistral-7B-Instruct-v0.2',
        model_kwargs=dict(tensor_parallel_size=1),
        max_out_len=256,
        batch_size=16,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=1),
    )
]
