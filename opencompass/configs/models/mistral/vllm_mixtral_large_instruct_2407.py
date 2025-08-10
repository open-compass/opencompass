from opencompass.models import VLLMwithChatTemplate


models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='mixtral-large-instruct-2407-vllm',
        path='mistralai/Mistral-Large-Instruct-2407',
        model_kwargs=dict(tensor_parallel_size=8),
        max_out_len=256,
        batch_size=16,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=8),
    )
]
